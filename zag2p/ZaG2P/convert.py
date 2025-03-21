#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @AUTHOR : thangdc94
# @Date : 2019/12/27
import os

from constant import project_root

if __name__ == '__main__':
    dict_path = os.path.join(project_root, "tts_dict_prepare/vn.dict")
    vietdict = {'6 b': '(bờ)', '6 k': '(cờ)', '6 tr': '(chờ)', '6 d': '(dờ)', '6 dd': '(đờ)', '6 g': '(gờ)',
                '6 l': '(lờ)', '6 m': '(mờ)', '6 n': '(nờ)', '6 p': '(pờ)', '6 ph': '(phờ)', '6 r': '(rờ)',
                '6 s': '(xờ)', '6 t': '(tờ)', '6 th': '(thờ)', '6 v': '(vờ)'}
    word_phone_map = inv_map = {v: k for k, v in vietdict.items()}
    with open(dict_path, "r", encoding="utf8") as fp:
        for line in fp.readlines():
            line = line.strip()
            if line:
                parts = line.split(" ", 1)
                word = parts[0]
                phoneme = parts[1]
                word_phone_map[word] = phoneme
    print(word_phone_map)
    input_file = "input.txt"
    output_file = "output.txt"
    with open(output_file, "w", encoding="utf8") as out_fp:
        with open(input_file, "r", encoding="utf8") as in_fp:
            for line in in_fp.readlines():
                line = line.strip()
                if line:
                    parts = line.split(" ", 1)
                    word = parts[0]
                    syllables = parts[1].split()
                    phonemes = " ".join([word_phone_map[x] for x in syllables])
                    out_fp.write("{} {}\n".format(word, phonemes))
