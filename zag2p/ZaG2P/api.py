# -*- coding: utf-8 -*-

""" Created on 10:30 AM, 9/4/19
    @author: ngunhuconchocon
    @brief: Пролетарии всех стран, соединяйтесь! да здравствует наша советская родина
"""
from __future__ import print_function, division, absolute_import

import argparse
import os
import re
from collections import OrderedDict

import bogo
import dill
import torch
import torchtext.data as data

from ZaG2P.DictClass import VNDict
from ZaG2P.constant import parser, project_root
from ZaG2P.models import G2P
from ZaG2P.utils import uncombine_phonemes_tone

tone_of_unvoiced_phoneme = "6"

accent_telex_dict = {
    '0': '', '1': 'f', '2': 'x', '3': 'r', '4': 's', '5': 'j'
}


def read_dict(dictpath):
    """
        this dict is different: dict[phoneme] = word, like dict['5 d i t'] = địt
    :param dictpath:
    :return:
    """
    vietdict = {'6 b': '(bờ)', '6 k': '(cờ)', '6 tr': '(chờ)', '6 d': '(dờ)', '6 dd': '(đờ)', '6 g': '(gờ)',
                '6 l': '(lờ)', '6 m': '(mờ)', '6 n': '(nờ)', '6 p': '(pờ)', '6 ph': '(phờ)', '6 r': '(rờ)',
                '6 s': '(xờ)', '6 t': '(tờ)', '6 th': '(thờ)', '6 v': '(vờ)'}
    with open(dictpath, "r", encoding="utf-8") as f:
        for line in f.readlines():
            if line and line.strip() and line[0] != "#":
                temp = line.strip().split(" ")
                word, phonemes = temp[0], temp[1:]

                vietdict[" ".join(phonemes)] = word
    return vietdict


def read_phoneme_syllable_vn_dict(phoneme_syllable_vn_dict_path):
    phoneme_syllable_vn_dict = {}
    with open(phoneme_syllable_vn_dict_path, 'r', encoding="utf-8") as fp:
        for cnt, line in enumerate(fp):
            parts = line.strip().split(" ≈ ")
            phonemes = parts[0]
            gram_count = len(phonemes.split())
            if gram_count not in phoneme_syllable_vn_dict:
                phoneme_syllable_vn_dict[gram_count] = {}
            phoneme_syllable_vn_dict[gram_count][parts[0]] = parts[1]
    my_dict = OrderedDict()
    for gram_dict in sorted(phoneme_syllable_vn_dict.items(), reverse=True):
        for k, v in gram_dict[1].items():
            my_dict[k] = v
    return my_dict


def convert_from_phonemes_to_syllables(batch, model, vietdict, phoneme_syllable_vn_dict):
    p_field = batch.dataset.fields['phoneme']
    prediction = model(batch.grapheme).tolist()[:-1]
    phonemes = ' '.join([p_field.vocab.itos[p] for p in prediction])
    uncombined_phonemes = uncombine_phonemes_tone(phonemes, None)
    prev = 0
    syllables = []

    for i, phoneme_or_tone in enumerate(uncombined_phonemes):
        if phoneme_or_tone.isdigit() and i > 1:
            combined_phonemes = " ".join(uncombined_phonemes[prev:i])
            syllables.append(convert_combined_phoneme(combined_phonemes, phoneme_syllable_vn_dict, vietdict))
            prev = i
        elif i == len(uncombined_phonemes) - 1:
            combined_phonemes = " ".join(uncombined_phonemes[prev:])
            syllables.append(convert_combined_phoneme(combined_phonemes, phoneme_syllable_vn_dict, vietdict))

    return " ".join(syllables)


def convert_combined_phoneme(combined_phonemes, phoneme_syllable_vn_dict, vietdict):
    if combined_phonemes in vietdict:
        return vietdict[combined_phonemes]
    else:
        accent_code = combined_phonemes[0]
        pattern = re.compile(r'\b(' + '|'.join(phoneme_syllable_vn_dict.keys()) + r')\b')
        result = pattern.sub(lambda x: phoneme_syllable_vn_dict[x.group()], combined_phonemes[1:])
        sequence = result.replace(" ", "") + accent_telex_dict[accent_code]
        new_seq = bogo.process_sequence(sequence)
        if len(new_seq) <= len(sequence):
            return new_seq
        else:
            return sequence


def load_model(model_path=None, dict_path=None, phoneme_syllable_vn_dict_path=None, use_cuda=True):
    args = argparse.Namespace(**parser)
    config = args

    if not model_path:
        model_path = os.path.join(project_root, os.path.join(config.intermediate_path, "best_model_adagrad_attn.pth"))
    if not dict_path:
        dict_path = os.path.join(project_root, "tts_dict_prepare/vn.dict")
    if not phoneme_syllable_vn_dict_path:
        phoneme_syllable_vn_dict_path = os.path.join(project_root, "tts_dict_prepare/phoneme_syllable_vn.dict")

    if use_cuda and torch.cuda.is_available():
        model_dict = torch.load(model_path, pickle_module=dill)
        p_field = model_dict['p_field']
        g_field = model_dict['g_field']
        config = model_dict['config']
        config.cuda = True
        model: G2P = G2P(config)
        model.load_state_dict(model_dict['model_state_dict'])
        model.cuda()
    else:
        model_dict = torch.load(model_path, map_location=torch.device('cpu'), pickle_module=dill)
        p_field = model_dict['p_field']
        g_field = model_dict['g_field']
        config = model_dict['config']
        config.cuda = False
        model = G2P(config)
        model.load_state_dict(model_dict['model_state_dict'])

    vietdict = read_dict(dictpath=dict_path)
    phoneme_syllable_vn_dict = read_phoneme_syllable_vn_dict(phoneme_syllable_vn_dict_path)
    return (g_field, p_field, model), vietdict, phoneme_syllable_vn_dict


def G2S(word, model_and_fields, vietdict, phoneme_syllable_vn_dict, use_cuda=True):
    """
        Convert grapheme to syllables
    :param word: string
    :param model_and_fields: model, getting from load_model(). Note that this contain g_field, p_field, and g2p model
    :param vietdict: vn.dict :D
    :param use_cuda:
    :return:
    """
    try:
        device = "cpu"
        if use_cuda:
            if torch.cuda.is_available():
                device = "cuda"
        word = word + " x x x"
        g_field, p_field, model = model_and_fields
        test_data = VNDict([word], g_field, p_field)
        test_iter = data.Iterator(test_data, batch_size=1, train=False, shuffle=True, device=torch.device(device))

        results = []
        for batch in test_iter:
            grapheme = batch.grapheme.squeeze(1).data.tolist()[1:][::-1]
            grapheme = ''.join([g_field.vocab.itos[g] for g in grapheme])
            results.append("{} {}".format(grapheme, convert_from_phonemes_to_syllables(batch, model, vietdict,
                                                                                       phoneme_syllable_vn_dict)))
            # print(results[-1])

        return results
    except:
        return word.split(" ")[0]
