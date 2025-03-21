#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @AUTHOR : thangdc94
# @Date : 2019/12/02
import atexit
import os
import signal

from vncorenlp import VnCoreNLP

from utils import tone_cleaner


class PhraseBreak:
    def __init__(self):
        print("Init VnCoreNLP")
        self.annotator = VnCoreNLP(os.path.abspath("VnCoreNLP/VnCoreNLP-1.1.1.jar"), annotators="wseg,pos,ner,parse",
                                   max_heap_size='-Xmx2g')
        print("Init VnCoreNLP success")
        # signal.signal(signal.SIGINT, self.cleanup)
        signal.signal(signal.SIGTERM, self.cleanup)

    def break_phrase(self, text: str):
        text = text.replace("_", "-")
        annotated_text = self.annotator.annotate(text)
        annotated_sentence = annotated_text['sentences'][0]
        phrase_break_list = []
        for annotated_word in annotated_sentence:
            head_annotated_word = annotated_sentence[annotated_word['head'] - 1]
            if (
                    (annotated_word['depLabel'] == 'nmod' and head_annotated_word['posTag'] in ['N', 'Np'])
                    or (annotated_word['depLabel'] in ['vmod', 'dob'] and head_annotated_word['posTag'] in ['V'])
                    or (annotated_word['depLabel'] == 'amod' and head_annotated_word['posTag'] in ['A'])
                    or (annotated_word['depLabel'] == 'adv' and head_annotated_word['posTag'] in ['A', 'V'])
                    or (annotated_word['depLabel'] == 'pob' and head_annotated_word['posTag'] in ['E'])
                    or (annotated_word['depLabel'] == 'det' and head_annotated_word['posTag'] in ['N', 'Np', 'M'])
                    or (annotated_word['depLabel'] in ['conj', 'coord'])
            ):
                if head_annotated_word['index'] > annotated_word['index']:
                    phrase_break_list.append(annotated_word['index'])
                else:
                    phrase_break_list.append(head_annotated_word['index'])

        new_text = []
        for annotated_word in annotated_sentence:
            new_text.append(tone_cleaner.clean_tone(annotated_word['form']))
            current_index = annotated_word['index']
            if current_index in phrase_break_list and (
                    current_index >= len(annotated_sentence)
                    or annotated_sentence[current_index]['depLabel'] != 'punct'
            ):
                new_text.append("-")
            else:
                new_text.append(" ")

        return "".join(new_text)

    def cleanup(self, signum, frame):
        print("Clean VnCoreNLP")
        self.annotator.close()
        exit()
