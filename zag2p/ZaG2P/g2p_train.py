#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @AUTHOR : thangdc94
# @Date : 2019/12/24
import argparse
import datetime
import os
import pickle
import time

import Levenshtein
import dill
import torch
from models import G2P
from torch import optim, nn

from DictClass import VNDict, CMUDict
from torchtext import data

from constant import parser, project_root


def adjust_learning_rate(optimizer, lr_decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay


def train(config, train_iter, model, criterion, optimizer, epoch):
    global iteration, n_total, train_loss, n_bad_loss
    global init, best_val_loss, stop

    print("=> EPOCH {}".format(epoch))
    train_iter.init_epoch()
    for batch in train_iter:
        iteration += 1
        model.train()

        output, _, __ = model(batch.grapheme, batch.phoneme[:-1].detach())
        target = batch.phoneme[1:]
        loss = criterion(output.view(output.size(0) * output.size(1), -1),
                         target.view(target.size(0) * target.size(1)))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), config.clip, 'inf')
        optimizer.step()

        n_total += batch.batch_size
        train_loss += loss.item() * batch.batch_size

        if iteration % config.log_every == 0:
            train_loss /= n_total
            val_loss = validate(val_iter, model, criterion)
            print("   % Time: {} | Dim {} | Iteration: {:5} | Batch: {:4}/{}"
                  " | Train loss: {:.4f} | Val loss: {:.4f}"
                  .format(datetime.timedelta(seconds=round(time.time() - init)), config.d_embed, iteration,
                          train_iter.iterations,
                          len(train_iter), train_loss, val_loss))

            # test for val_loss improvement
            n_total = train_loss = 0
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                n_bad_loss = 0
                torch.save({"model_state_dict": model.state_dict(),
                            "p_field": p_field,
                            "g_field": g_field,
                            "config": config,
                            "train_data": all_data
                            },
                           config.best_model
                           , pickle_module=dill)
            else:
                n_bad_loss += 1
            if n_bad_loss == config.n_bad_loss:
                best_val_loss = val_loss
                n_bad_loss = 0
                adjust_learning_rate(optimizer, config.lr_decay)
                new_lr = optimizer.param_groups[0]['lr']
                print("=> Adjust learning rate to: {}".format(new_lr))
                if new_lr < config.lr_min:
                    stop = True
                    break


def validate(val_iter, model, criterion):
    model.eval()
    val_loss = 0
    val_iter.init_epoch()
    for batch in val_iter:
        output, _, __ = model(batch.grapheme, batch.phoneme[:-1])
        target = batch.phoneme[1:]
        loss = criterion(output.squeeze(1), target.squeeze(1))
        val_loss += loss.item() * batch.batch_size
    return val_loss / len(val_iter.dataset)


def phoneme_error_rate(p_seq1, p_seq2):
    p_vocab = set(p_seq1 + p_seq2)
    p2c = dict(zip(p_vocab, range(len(p_vocab))))
    c_seq1 = [chr(p2c[p]) for p in p_seq1]
    c_seq2 = [chr(p2c[p]) for p in p_seq2]
    return Levenshtein.distance(''.join(c_seq1),
                                ''.join(c_seq2)) / len(c_seq2)


def test(test_iter, model, criterion):
    model.eval()
    test_iter.init_epoch()
    test_per = test_wer = 0
    for batch in test_iter:
        output = model(batch.grapheme).data.tolist()
        target = batch.phoneme[1:].squeeze(1).data.tolist()
        # calculate per, wer here
        per = phoneme_error_rate(output, target)
        wer = int(output != target)
        test_per += per  # batch_size = 1
        test_wer += wer

    test_per = test_per / len(test_iter.dataset) * 100
    test_wer = test_wer / len(test_iter.dataset) * 100
    return test_per, test_wer


def show(batch, model):
    assert batch.batch_size == 1
    g_field = batch.dataset.fields['grapheme']
    p_field = batch.dataset.fields['phoneme']
    prediction = model(batch.grapheme).data.tolist()[:-1]
    grapheme = batch.grapheme.squeeze(1).data.tolist()[1:][::-1]
    phoneme = batch.phoneme.squeeze(1).data.tolist()[1:-1]
    print("> {}\n= {}\n< {}\n".format(
        ''.join([g_field.vocab.itos[g] for g in grapheme]),
        ' '.join([p_field.vocab.itos[p] for p in phoneme]),
        ' '.join([p_field.vocab.itos[p] for p in prediction])))


if __name__ == '__main__':
    args = argparse.Namespace(**parser)
    config = args
    g_field = data.Field(init_token='<s>', tokenize=(lambda x: list(x.split()[0])[::-1]))
    p_field = data.Field(init_token='<os>', eos_token='</os>', tokenize=(lambda x: x.split()))

    filepath = os.path.join(project_root, os.path.join("tts_dict_prepare", 'oov.vn.dict'))
    train_data, val_data, test_data, all_data = VNDict.splits(filepath, g_field, p_field, config.seed, drop_headline=0)
    if config.cuda:
        device = "cuda"
    else:
        device = "cpu"
    train_iter = data.BucketIterator(train_data, batch_size=args.batch_size,
                                     repeat=False, device=device)
    val_iter = data.Iterator(val_data, batch_size=1,
                             train=False, sort=False, device=device)
    test_iter = data.Iterator(test_data, batch_size=1,
                              train=False, shuffle=True, device=device)

    g_field.build_vocab(train_data, val_data, test_data)
    p_field.build_vocab(train_data, val_data, test_data)
    config.g_size = len(g_field.vocab)
    config.p_size = len(p_field.vocab)
    per_dict = {}
    wer_dict = {}
    for dim in range(200, 700, 50):
        config.d_embed = dim
        config.d_hidden = dim
        config.best_model = os.path.join(config.intermediate_path,
                                         "best_model_adagrad_attn_{}.pth".format(dim))
        model = G2P(config)
        criterion = nn.NLLLoss()
        if config.cuda:
            model.cuda()
            criterion.cuda()
        optimizer = optim.Adagrad(model.parameters(), lr=config.lr)  # use Adagrad
        if 1 == 1:  # change to True to train
            iteration = n_total = train_loss = n_bad_loss = 0
            stop = False
            best_val_loss = 10
            init = time.time()
            for epoch in range(1, config.epochs + 1):
                train(config, train_iter, model, criterion, optimizer, epoch)
                if stop:
                    break
        print("==================== TEST ==================")
        test_per, test_wer = test(test_iter, model, criterion)
        print("Phoneme error rate (PER): {:.2f}\nWord error rate (WER): {:.2f}"
              .format(test_per, test_wer))
        per_dict[dim] = test_per
        wer_dict[dim] = test_wer
    with open("result_er.pkl", "wb") as fp:
        pickle.dump((per_dict, wer_dict), fp)
    test_iter.init_epoch()
    for i, batch in enumerate(test_iter):
        show(batch, model)
        if i == 10:
            break
