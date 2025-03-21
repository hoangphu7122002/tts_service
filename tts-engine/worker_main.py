#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @AUTHOR : thangdc94
# @Date : 2021/10/12
import argparse
import os

from celery.utils.nodenames import gethostname
from shortuuid import ShortUUID

import config
from model_task_queue import mycelery
from model_task_queue.tasks import SynthesizeVoiceTask

if os.name == 'nt':
    os.environ.setdefault('FORKED_BY_MULTIPROCESSING', '1')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--voice', required=True, choices=list(config.TACOTRON_PATHS.keys()))
    parser.add_argument('--cuda', default='0')
    parser.add_argument('--new', default=False, type=bool)
    opt = parser.parse_args()
    queue = config.QUEUES[opt.voice]
    mycelery.app.register_task(SynthesizeVoiceTask(opt.voice, opt.cuda, opt.new))
    worker_config = {'concurrency': 1, 'task_events': True, 'queues': [queue], 'loglevel': "INFO"}
    w = mycelery.app.Worker(hostname=f"{opt.voice}-{ShortUUID().random(4)}@{gethostname()}", **worker_config)
    w.start()


if __name__ == '__main__':
    main()
