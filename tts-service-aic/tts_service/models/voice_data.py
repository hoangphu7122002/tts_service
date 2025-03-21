#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @AUTHOR : thangdc94
# @Date : 2020/01/18


class VoiceData:
    def __init__(self, retry: int = None, timeout_ms: int = None, url: str = None):
        self.retry = retry
        self.timeoutMilSecs = timeout_ms
        self.url = url
