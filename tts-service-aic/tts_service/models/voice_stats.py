#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @AUTHOR : thangdc94
# @Date : 2020/01/18


class VoiceStats:
    def __init__(self, num_words: int = None, time: float = None, words_per_second: float = None):
        self.num_words = num_words
        self.time = time
        self.words_per_second = words_per_second
