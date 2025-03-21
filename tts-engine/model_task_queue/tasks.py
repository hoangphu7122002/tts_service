#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @AUTHOR : thangdc94
# @Date : 2021/10/12
import logging
import os
import re

import numpy as np
from celery import Task

import config
from phrase_break import PhraseBreak
from tts_model import VoiceEnd2End, TtsDecodeError

logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s', level=logging.INFO)


def split_long_sentence(text, max_words):
    result = []
    for sub_sen in text.strip().split(','):
        sub_sen = sub_sen.strip()
        tokens = []
        count = 0
        for word in sub_sen.split():
            grams = re.split("_|-", word)
            n_gram = len(grams)
            if count + n_gram > max_words:
                count = 0
                tokens.append(",")
                result.append(' '.join(tokens))
                tokens = []
            count += n_gram
            tokens.extend(grams)

        result.append(' '.join(tokens))

    text = ','.join(result)
    result = []
    sen = ""
    for sub_sen in text.strip().split(','):
        sub_sen = sub_sen.strip()
        if len((sen + " " + sub_sen).split()) > max_words:
            result.append(sen)
            sen = ""
        if len(sen) > 0:
            sen += " , "
        sen += sub_sen
    if len(sen) > 0:
        result.append(sen)
    return result


class SynthesizeVoiceTask(Task):
    TASK_NAME = "synthesize_voice"
    name = f"{__name__}.{TASK_NAME}"
    autoretry_for = (TtsDecodeError,)
    retry_backoff = True
    retry_backoff_max = 1

    def __init__(self, voice, cuda, use_new_cfg) -> None:
        super().__init__()
        self.__voice = voice
        self.__cuda = cuda
        self.__use_new_cfg = use_new_cfg
        self.__max_words = config.MAX_WORDS[self.__voice]
        self.__tts_model = None
        self.__phrase_break = PhraseBreak()

    def initialize_model(self):
        """
        Cold start scenario for each worker process
        Load model on first call (i.e. first task processed)
        Avoids the need to load model on each task request
        Cannot load model in constructor because we need to create TTS model for each worker
        """
        if not self.__tts_model:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.__cuda
            use_eos = config.USE_EOS[self.__voice]
            self.__tts_model = VoiceEnd2End(voice=self.__voice, new=self.__use_new_cfg, use_eos=use_eos)

    def run(self, data):
        self.initialize_model()
        utt = data['utt']
        text = data['text']
        logging.info("Receive {}: {}".format(utt, text))
        audio_out = []
        for sen in text.split('.'):
            sen = sen.strip()
            if len(sen) > 0:
                phrase_broken_sentence = self.__phrase_break.break_phrase(sen)
                logging.info("Phrase Break: {}".format(phrase_broken_sentence))
                for sub_sen in split_long_sentence(phrase_broken_sentence, self.__max_words):
                    sub_sen = sub_sen.replace("_", " ").replace("-", " ").lower()
                    sub_sen = sub_sen.strip().strip(',').strip()
                    logging.info("Segment: {}".format(sub_sen))
                    audio = self.__tts_model.speech(sub_sen)
                    if audio is not None:
                        audio_out += audio.tolist() + [0] * int(0.1 * self.__tts_model.sampling_rate)
                    else:
                        raise RuntimeError("Failed to synthesize Audio")
                audio_out += [0] * int(0.25 * self.__tts_model.sampling_rate)
        audio_out = np.array(audio_out)
        return dict(
            utt=utt,
            audio_data=audio_out.tolist(),
            sampling_rate=self.__tts_model.sampling_rate,
            is_success=True
        )
