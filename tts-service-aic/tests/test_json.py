#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @AUTHOR : thangdc94
# @Date : 2020/01/18

import json

from tts_service.constants import return_codes
from tts_service.models.voice_data import VoiceData
from tts_service.models.voice_result import VoiceResult
from tts_service.models.voice_stats import VoiceStats

# s = '{"status":111,"msg":"Failed to join file!"}'
# o: VoiceResult = json.loads(s, object_hook=VoiceResult.from_dict)

voice_data = VoiceData(url=".mp3")

voice_stats = VoiceStats(num_words=10, time=2, words_per_second=100)
obj: VoiceResult = VoiceResult(status=return_codes.API_SUCCESS,
                               msg="Success!",
                               data=voice_data,
                               stats=voice_stats)
print(obj.__dict__)

json_data = json.dumps(obj, default=lambda o: o.__dict__)
print(json_data)
