#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @AUTHOR : thangdc94
# @Date : 2020/01/18
from .base_response import BaseResponse
from .voice_data import VoiceData
from .voice_stats import VoiceStats


class VoiceResult(BaseResponse):

    def __init__(self, msg=None, status=None, data: VoiceData = None, stats: VoiceStats = None):
        super().__init__(msg, status)
        self.data = data
        self.stats = stats

    @staticmethod
    def from_dict(d):
        instance = VoiceResult()
        instance.__dict__.update(d)
        return instance
