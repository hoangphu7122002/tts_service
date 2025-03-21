import contextlib
import hashlib
import json
import os
import re
import socket
import subprocess
import time
import traceback
import uuid
import wave
from datetime import datetime
from typing import Sequence

import celery
import numpy as np
import requests
from celery import group
from celery.result import AsyncResult, GroupResult
from flask_api import status

from tts_service.constants import return_codes
from tts_service.models.base_response import BaseResponse
from tts_service.models.voice_data import VoiceData
from tts_service.models.voice_result import VoiceResult
from tts_service.models.voice_stats import VoiceStats
from tts_service.utils import logutils, jsonutils
from tts_service.utils.audio_utils import save_audio
from . import config
from . import mycelery

logger = logutils.getLogger(__name__)


class Voice:
    def speak(self, text):
        pass


class VoiceEnd2End(Voice):
    def __init__(self, voice, request_queue, k_timeout=0.12, k_abnormal=0.3):
        self.voice = voice
        self.request_queue = request_queue
        self.k_timeout = k_timeout
        self.k_abnormal = k_abnormal
        self.data_dir = os.path.join(config.STORAGE_DATA, self.voice)
        self.chunk_dir = os.path.join(config.STORAGE_CHUNK, self.voice)

        if not os.path.isdir(os.path.join(config.SITE_ROOT, self.data_dir)):
            os.makedirs(os.path.join(config.SITE_ROOT, self.data_dir))
        if not os.path.isdir(os.path.join(config.SITE_ROOT, self.chunk_dir)):
            os.makedirs(os.path.join(config.SITE_ROOT, self.chunk_dir))

    @staticmethod
    def get_duration(wav_path):
        try:
            with contextlib.closing(wave.open(wav_path, 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)
                return duration
        except:
            logger.error(wav_path)
            logger.error(traceback.format_exc())
            return 10000000

    @staticmethod
    def norm_text(idx, text, domain, normOOV=False, keepOovToken=False):
        data = dict()
        if normOOV:
            data['normOov'] = 'true'
        else:
            data['normOov'] = 'false'
        data['content'] = text
        data['domain'] = domain
        if keepOovToken:
            data['keepOovToken'] = 'true'
        else:
            data['keepOovToken'] = 'false'
        try:
            response = requests.post(config.NORM_URL, data=data,
                                     headers={'Content-Type': "application/x-www-form-urlencoded; charset=UTF-8"})
            if response.status_code != 200:
                text_norm = text
                logger.error("{} Cannot connect to normalize service!".format(idx))
                logger.error("{} {}".format(idx, response))
            else:
                result = json.loads(response.text)
                text_norm = result['normText']
                logger.info("{} Normalize: {}".format(idx, text_norm))
        except:
            logger.error("{} Cannot connect to normalize service!".format(idx))
            text_norm = text.lower()
        return text_norm

    def speak_rabbitMQ(self, domain, text, caching=True, normOOV=False):
        # Get start time for checking timeout
        start_time = time.time()
        idx = datetime.now().strftime("%Y%m%d%H%M%S") + "-" + str(uuid.uuid4())[:8]
        logger.info("{} Receive: {}".format(idx, text))

        # norm text here
        text_norm = self.norm_text(idx, text, domain, normOOV, True)
        list_audios = []
        processing_part = []
        list_words_count = []
        list_message = []

        # Split text and send jobs to rabbitMQ
        for sentence in text_norm.strip().split('.'):
            sentence = sentence.strip()
            if len(sentence) <= 1:
                continue
            subname = hashlib.md5(sentence.encode('utf-8')).hexdigest()
            logger.info("{} Hashed {} | {}".format(idx, subname, sentence))
            audio_path = os.path.join(os.path.join(config.SITE_ROOT, self.chunk_dir), "{}.wav".format(subname))
            processing_part.append(subname)
            list_audios.append(audio_path)
            # list_words_count.append(len(sentence.replace(",", "").split()))
            list_words_count.append(len(re.split(" |_|-", sentence)))
            if caching and os.path.exists(audio_path):
                logger.info("{} Cached {}".format(idx, audio_path))
            else:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                list_message.append((subname, sentence))
        total_words = sum(list_words_count)
        timeout_time = max(config.MIN_TIMEOUT, self.k_timeout * total_words)

        # Connect RabitMQ
        all_request = []
        for subname, sentence in list_message:
            req_rabbit_mq = {
                'utt': subname,
                'text': sentence,
            }
            all_request.append(
                mycelery.app.signature("model_task_queue.tasks.synthesize_voice", args=(req_rabbit_mq,), expires=timeout_time)
            )
        g = group(all_request)
        group_result: GroupResult = g.apply_async(queue=self.request_queue)
        logger.info("{} Send task group {} to queue {}".format(idx, group_result.id, self.request_queue))
        # TODO: make progress bar here
        try:
            results = group_result.get(timeout=timeout_time)
            group_result.forget()
            for result in results:
                utt = result['utt']
                # logger.info("Received {} -> Success: {}".format(utt, is_success))
                audio_data = result['audio_data']
                sampling_rate = result['sampling_rate']
                if utt in processing_part:
                    _wav_path = os.path.join(os.path.join(config.SITE_ROOT, self.chunk_dir), "{}.wav".format(utt))
                    save_audio(np.array(audio_data), _wav_path, sampling_rate)
                    processing_part.remove(utt)
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.info("Revoke all subtasks")
            group_result.revoke()
            if (isinstance(e, celery.exceptions.TaskRevokedError) and str(e) == 'expired') \
                    or isinstance(e, celery.exceptions.TimeoutError):
                logger.error("{} Timeout ({} words - {}s)!".format(idx, total_words, timeout_time))
                return jsonutils.to_json(BaseResponse(status=return_codes.API_CODE_TIMEOUT,
                                                      msg="Timeout!")), status.HTTP_500_INTERNAL_SERVER_ERROR
            return jsonutils.to_json(BaseResponse(status=return_codes.API_CANNOT_READ,
                                                  msg="TTS cannot read the text!")), status.HTTP_500_INTERNAL_SERVER_ERROR

        logger.info("Received all parts of {}".format(idx))

        wav_path = os.path.join(config.SITE_ROOT, self.data_dir, idx + ".wav")
        mp3_path = os.path.join(config.SITE_ROOT, self.data_dir, idx + ".mp3")
        cmd = "sox {lists} {wav} > /dev/null 2>&1 && ffmpeg -y -i {wav} {mp3} > /dev/null 2>&1".format(
            lists=' '.join(list_audios),
            wav=wav_path,
            mp3=mp3_path
        )
        p = subprocess.Popen(cmd, shell=True)
        stdout, stderr = p.communicate()
        if p.returncode == 0:
            logger.info("Joined all parts of {}".format(idx))
            try:
                for audio_file in list_audios:
                    os.remove(audio_file)
                os.remove(wav_path)
                logger.info("Deleted all temp parts")
            except IOError:
                logger.exception("Delete Error")
            logger.info("{} Created {}".format(idx, mp3_path))
            voice_data = VoiceData(url="{}/{}".format(config.DOMAIN, os.path.join(self.data_dir, idx + ".mp3")))
            voice_stats = VoiceStats(num_words=total_words, time=time.time() - start_time,
                                     words_per_second=sum(list_words_count) / (time.time() - start_time))
            return jsonutils.to_json(VoiceResult(status=return_codes.API_SUCCESS,
                                                 msg="Success!",
                                                 data=voice_data,
                                                 stats=voice_stats))

        logger.error("{} Join file failed!".format(idx))
        return jsonutils.to_json(BaseResponse(status=return_codes.API_ERROR,
                                              msg="Failed to join file!")), status.HTTP_500_INTERNAL_SERVER_ERROR

    @staticmethod
    def wait_tasks_ready(results: Sequence[AsyncResult], timeout_time: int):
        time_start = time.monotonic()
        while True:
            if time.monotonic() - time_start >= timeout_time:
                raise socket.timeout()
            completed_count = 0
            for result in results:
                if result.ready():
                    completed_count += 1
                if result.failed():
                    result.get()
            if completed_count == len(results):
                break
