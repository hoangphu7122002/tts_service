#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @AUTHOR : thangdc94
# @Date : 2019/12/30
import argparse
import os
import pickle

import Levenshtein
import torch
from torch import nn
from torchtext import data

from DictClass import VNDict
from api import load_model
from constant import project_root, parser


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
    print("==================== TEST ==================")
    args = argparse.Namespace(**parser)
    config = args
    if config.cuda:
        device = "cuda"
    else:
        device = "cpu"
    per_dict = {}
    wer_dict = {}
    for dim in range(200, 750, 50):
        model_path = config.intermediate_path + "best_model_adagrad_attn_{}.pth".format(dim)
        print(model_path)
        (g_field, p_field, model), _, _ = load_model(
            model_path=model_path,
            use_cuda=config.cuda)

        filepath = os.path.join(project_root, os.path.join("tts_dict_prepare", 'oov.vn.dict'))
        train_data, val_data, test_data, all_data = VNDict.splits(filepath, g_field, p_field, seed=config.seed,
                                                                  drop_headline=0)

        test_iter = data.Iterator(test_data, batch_size=1,
                                  train=False, shuffle=True, device=device)

        criterion = nn.NLLLoss()

        test_per, test_wer = test(test_iter, model, criterion)
        print("Phoneme error rate (PER): {:.2f}\nWord error rate (WER): {:.2f}"
              .format(test_per, test_wer))
        per_dict[dim] = test_per
        wer_dict[dim] = test_wer
        test_iter.init_epoch()
        for i, batch in enumerate(test_iter):
            show(batch, model)
            if i == 10:
                break
        with open("result_er.pkl", "wb") as fp:
            pickle.dump((per_dict, wer_dict), fp)

