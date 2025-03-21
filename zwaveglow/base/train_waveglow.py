import argparse
import json
import os
import torch
import sys
from tensorboardX import SummaryWriter   
from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from glow import WaveGlow, WaveGlowLoss
from mel2samp import Mel2Samp
import time
import glob
from paths import Paths
from utils.display import simple_table
from apex import amp

def print_log(text=""):
    sys.stdout.write(text + "\n")
    sys.stdout.flush()

def load_checkpoint(checkpoint_path, model, optimizer):
    print_log("\nLoading model from checkpoint '{}'".format(
        checkpoint_path))
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    model_for_loading = checkpoint_dict['model']
    epoch = checkpoint_dict['epoch']
    learning_rate = checkpoint_dict['learning_rate']
    model.load_state_dict(model_for_loading.state_dict())
    return model, optimizer, iteration, epoch, learning_rate

def save_checkpoint(model, init_model,
                    optimizer, learning_rate, iteration, epoch, filepath):
    print_log("Saving to {}".format(filepath))   
    init_model.load_state_dict(model.state_dict())
    torch.save({'model': init_model,
                'iteration': iteration,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate,           
                'epoch': epoch}, filepath)

def train(num_gpus, rank, group_name, config_file_path): 
    with open(config_file_path) as f:
        lines = f.read()
    config = json.loads(lines) 
    training_config = config["training_config"]
    feature_config = config["feature_config"]
    optimizer_config = config["optimizer_config"]
    
    if num_gpus > 1:
        dist_backend = "nccl"
        dist_url = "tcp://localhost:54321"
        init_distributed(rank, num_gpus, group_name, dist_backend, dist_url)
  
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
  
    if rank == 0: 
        paths = Paths(config["data"], config["version"])
        paths.create_training_paths()
        logger = SummaryWriter(paths.logs_dir_path)
               
    criterion = WaveGlowLoss(optimizer_config["sigma"])
    init_model = WaveGlow(feature_config["n_mel_channels"], feature_config["filter_length"], feature_config["hop_length"], **config["model_config"]).cuda()
    
    model = init_model
  
    if num_gpus > 1:
        model = apply_gradient_allreduce(init_model)

    optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_config["learning_rate"])

    if training_config["fp16_run"]:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    
    simple_table([('data', config["data"]),
                  ('version', config["version"]),
                  ('pre_emphasize', config["audio_config"]["pre_emphasize"]),
                  ('rescale', config["audio_config"]["rescale"]),
                  ('sampling_rate', feature_config["sampling_rate"])])

    iteration = 0
    epoch_offset = 0
    init_learning_rate = optimizer_config["learning_rate"]
    checkpoint_path = training_config["checkpoint_path"]     
    if checkpoint_path == "last":   
        part1 = 'waveglow_' + os.path.basename(config["data"]) + '_' + config["version"] + '_'
        part3 = 'k.pt'
        part2s = [int(chp.replace(part1, '').replace(part3, '')) for chp in os.listdir(paths.checkpoints_dir_path) if chp.startswith(part1)]  
        if part2s:
            checkpoint_path = os.path.join(paths.checkpoints_dir_path, part1 + str(max(part2s))) + part3
    if os.path.isfile(checkpoint_path):
        model, optimizer, iteration, epoch_offset, _learning_rate_last = load_checkpoint(checkpoint_path, model,
                                                                          optimizer)
        iteration += 1    
        #epoch_offset += 1
    else:
        print_log('\nTraining from scratch')     

    trainset = Mel2Samp(paths.train_list_file_path, **config["audio_config"], **feature_config)
  
    train_sampler = DistributedSampler(trainset) if num_gpus > 1 else None
  
    train_loader = DataLoader(trainset, num_workers=1, shuffle=False,
                              sampler=train_sampler,
                              batch_size=training_config["batch_size"],
                              pin_memory=False,
                              drop_last=True)

    model.train()
    model_last = None
    iteration_last = None
    optimizer_last = None
    learning_rate_last = None
    # ================ MAIN TRAINNIG LOOP! ===================  
    for epoch in range(epoch_offset, training_config["epochs"]):   
        for i, batch in enumerate(train_loader):
            try:
                start = time.perf_counter()
                start_real = time.time()
                learning_rate = init_learning_rate*(optimizer_config["lr_decay"] ** (iteration // optimizer_config["iters_per_decay_lr"]))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate
                model.zero_grad()
                mel, audio = batch
                mel = torch.autograd.Variable(mel.cuda())
                audio = torch.autograd.Variable(audio.cuda())
                outputs = model((mel, audio))
                loss = criterion(outputs)
                if num_gpus > 1:
                    reduced_loss = reduce_tensor(loss.data, num_gpus).item()
                else:
                    reduced_loss = loss.item()
                if training_config["fp16_run"]:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optimizer.step()
                model_last = model
                iteration_last = iteration
                optimizer_last = optimizer
                learning_rate_last = learning_rate
                
                speed = time.perf_counter() - start      
                speed_real = time.time() - start_real                              
                print_log("Epoch: {}   Iter: {}   Loss: {:.6f}   Speed: {:.2f}s/it   Speed-real: {:.2f}s/it".format(epoch, iteration, reduced_loss, speed, speed_real))
                if rank == 0:
                    logger.add_scalar('training_loss', reduced_loss, iteration)
                    logger.add_scalar('learning_rate', learning_rate, iteration)
                    #logger.add_scalar('batch_size', batch_size*num_gpus, iteration)    
                    #logger.add_scalar('fp16_run', int(fp16_run), iteration)  
                    #logger.add_scalar('training_speed', speed, iteration)
                    #logger.add_scalar('training_sigma', sigma, iteration)   
                    
                if (iteration % training_config["iters_per_checkpoint"] == 0) and rank == 0:                               
                    checkpoint_path = f"{paths.checkpoints_dir_path}/waveglow_{os.path.basename(config['data'])}_{config['version']}_{iteration // 1000}k.pt"
                    save_checkpoint(model, init_model, optimizer, learning_rate, iteration, epoch, checkpoint_path)
                                                 
                iteration += 1
                             
            except KeyboardInterrupt:
                print_log()              
                checkpoint_path = f"{paths.checkpoints_dir_path}/waveglow_{os.path.basename(config['data'])}_{config['version']}_{iteration // 1000}k.pt"              
                save_checkpoint(model_last, init_model, optimizer_last, learning_rate_last, iteration_last, epoch, checkpoint_path)
                print_log('Stop Training')
                os._exit(1)
       

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.json',
                        help='JSON file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='',
                        help='name of group for distributed')
    parser.add_argument('-d', '--cuda', type=str)
    args = parser.parse_args()
    
    if not args.cuda:
        args.cuda = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        if args.group_name == '':
            print_log("WARNING: Multiple GPUs detected but no distributed group set")
            print_log("Only running 1 GPU.  Use distributed.py for multiple GPUs")
            num_gpus = 1
    if num_gpus == 1 and args.rank != 0:
        raise Exception("Doing single GPU training on rank > 0")
        
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
       
    train(num_gpus, args.rank, args.group_name, args.config)
