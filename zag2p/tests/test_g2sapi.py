#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @AUTHOR : thangdc94
# @Date : 2019/12/12
import time
from unittest import TestCase

from api import load_model, G2S


class TestG2sApi(TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)

    def test_g2s_readable(self):
        model, vietdict, phoneme_syllable_vn_dict = load_model(use_cuda=False)

        with open("test_set.txt") as fp:
            for line in fp.readlines():
                word = line.strip()
                syllable = G2S(line.lower(), model, vietdict, phoneme_syllable_vn_dict, use_cuda=False)
                print("{} -> {}".format(word, syllable))
