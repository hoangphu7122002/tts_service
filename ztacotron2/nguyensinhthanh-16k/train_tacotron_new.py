import os
import sys
import time
import argparse
import math
from numpy import finfo
from utils import simple_table
import torch
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from model import Tacotron2, Tacotron2Loss
from data_utils import TextMelLoader, TextMelCollate
from logger import Tacotron2Logger
from hparams import create_hparams_and_paths
from text_embedding import word2phone, symbol2numeric
import glob
import warnings
warnings.filterwarnings("ignore")

def print_log(text=""):
    sys.stdout.write(text + "\n")
    sys.stdout.flush()

def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= n_gpus
    return rt


def init_distributed(n_gpus, rank, group_name, dist_backend="nccl", dist_url="tcp://localhost:12345"):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print_log("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=dist_backend, init_method=dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)

    print_log("Done initializing distributed")


def prepare_dataloaders(hparams, training_files, validation_files):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(training_files, hparams)
    valset = TextMelLoader(validation_files, hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(trainset, num_workers=1, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    return train_loader, valset, collate_fn


def load_model(hparams):
    model = Tacotron2(hparams).cuda()
    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    return model


def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print_log("Warm starting model from checkpoint {}\n".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    print_log("Loading model from check point {}\n".format(checkpoint_path))
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    epoch = checkpoint_dict['epoch']
    iteration = checkpoint_dict['iteration']
    return model, optimizer, learning_rate, epoch, iteration


def save_checkpoint(model, hparams, optimizer, learning_rate, epoch, iteration, filepath):
    print_log("Saving to {}".format(filepath))
    vn2phone_train_dict = word2phone(hparams.phone_vn_train, hparams.coda_nucleus_and_semivowel)
    oov2phone_train_dict = word2phone(hparams.phone_oov_train, hparams.coda_nucleus_and_semivowel)
    symbol2numeric_dict = symbol2numeric(hparams)
    torch.save({'iteration': iteration,
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'hparams': hparams,
                'vn2phone_train_dict': vn2phone_train_dict,
                'oov2phone_train_dict': oov2phone_train_dict,
                'symbol2numeric_dict': symbol2numeric_dict,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate},
                  filepath)


def validate(model, criterion, valset, iteration, batch_size, n_gpus, collate_fn, logger, distributed_run, rank):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=1,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)
        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
        val_loss = val_loss / (i + 1)

    model.train()
    if rank == 0:
        print_log("Validation loss: {:9f}  ".format(reduced_val_loss))
        logger.log_validation(reduced_val_loss, y, y_pred, iteration)


def train(n_gpus, rank, group_name, hp, paths):


    if hp.distributed_run:
        init_distributed(n_gpus, rank, group_name)

    torch.manual_seed(seed=1234)
    torch.cuda.manual_seed(seed=1234)

    models_directory = paths.tacotron_models_dir_path
    logs_directory = paths.logs_dir_path
    if rank == 0:
        paths.create_training_paths()
        logger = Tacotron2Logger(logs_directory)
    else:
        logger = None

    # prepare model
    model = load_model(hp)
    ini_learning_rate = hp.init_lr
    optimizer = torch.optim.Adam(model.parameters(), lr=ini_learning_rate, eps=hp.eps, weight_decay=hp.weight_decay)
    if hp.fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O2')
    if hp.distributed_run:
        model = apply_gradient_allreduce(model)
    criterion = Tacotron2Loss()

    train_loader, valset, collate_fn = prepare_dataloaders(hp, paths.train_list_file_path, paths.val_list_file_path)

    simple_table([('data', hp.data),
                  ('version', hp.version),
                  ('warm_start', hp.warm_start),
                  ('p_phone_mix', hp.p_phone_mix),
                  ('sampling_rate', hp.sampling_rate)])

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    checkpoint_path = hp.checkpoint_path
    if checkpoint_path != "":
        if checkpoint_path == 'last' and len(glob.glob(os.path.join(models_directory, '*'))) > 0:
            part1 = 'tacotron2_'
            part3 = 'k.pt'
            part2s = [int(chp.replace(part1, '').replace(part3, '')) for chp in os.listdir(models_directory) if
                      chp.startswith(part1)]
            checkpoint_path = os.path.join(models_directory, part1 + str(max(part2s))) + part3
        if hp.warm_start:
            model = warm_start_model(checkpoint_path, model, hp.ignore_layers)
        else:
            model, optimizer, _learning_rate_last, epoch_offset, iteration = load_checkpoint(checkpoint_path, model,
                                                                                             optimizer)
            iteration += 1
            if hp.use_last_lr and _learning_rate_last >= hp.final_lr:
                ini_learning_rate = _learning_rate_last


    model.train()
    is_overflow = False
    # ================ MAIN TRAINNIG LOOP! ===================
    model_last = None
    iteration_last = None
    optimizer_last = None
    learning_rate_last = None
    learning_rate = ini_learning_rate
    for epoch in range(epoch_offset, hp.epochs):
        for i, batch in enumerate(train_loader):
            try:
                start = time.perf_counter()
                if iteration >= hp.iter_start_decay:
                    learning_rate = ini_learning_rate * (hp.lr_decay ** (iteration // hp.iters_per_decay_lr))
                learning_rate = max(hp.final_lr, learning_rate)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate

                model.zero_grad()
                x, y = model.parse_batch(batch)
                y_pred = model(x)

                loss = criterion(y_pred, y)
                if hp.distributed_run:
                    reduced_loss = reduce_tensor(loss.data, n_gpus).item()
                else:
                    reduced_loss = loss.item()
                if hp.fp16_run:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                if hp.fp16_run:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), hp.grad_clip_thresh)
                    is_overflow = math.isnan(grad_norm)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip_thresh)
                optimizer.step()

                model_last = model
                iteration_last = iteration
                optimizer_last = optimizer
                learning_rate_last = learning_rate

                if not is_overflow and rank == 0:
                    speed = time.perf_counter() - start
                    print_log("Epoch: {}   Iter: {}   Loss: {:.6f}   Grad: {:.6f}   Speed: {:.3f}s/it".format(epoch, iteration, reduced_loss, grad_norm, speed))
                    logger.log_training(reduced_loss, learning_rate, iteration)

                if not is_overflow:
                    if (iteration % hp.iters_per_valid == 0):
                        validate(model, criterion, valset, iteration, hp.batch_size, n_gpus, collate_fn,
                                 logger, hp.distributed_run, rank)

                    if rank == 0 and (iteration % hp.iters_per_checkpoint == 0) and iteration > 0:
                        checkpoint_path = os.path.join(models_directory, "tacotron2_{}k.pt".format(iteration // 1000))
                        save_checkpoint(model, hp, optimizer, learning_rate, epoch, iteration, checkpoint_path)
                iteration += 1

            except KeyboardInterrupt:
                print_log()
                checkpoint_path = os.path.join(models_directory, "tacotron2_{}k.pt".format(iteration // 1000))
                save_checkpoint(model_last, hp, optimizer_last, learning_rate_last, epoch, iteration_last, checkpoint_path)
                print_log('Stop Training')
                os._exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', '-p', type=str,
                        required=False, help='comma separated name=value pairs')
    parser.add_argument('--cuda', '-c', type=str, default='0',
                        required=False)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    hp, paths = create_hparams_and_paths(args.hparams)
    train(args.n_gpus, args.rank, args.group_name, hp, paths)
