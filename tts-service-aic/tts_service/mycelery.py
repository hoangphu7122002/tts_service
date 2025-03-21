#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @AUTHOR : thangdc94
# @Date : 2021/10/11
from celery import Celery

app = Celery(
    'celery_tts_app',
    # broker='amqp://guest:guest@10.240.187.16:5672//',
    # backend='rpc://',
    broker='redis://10.240.187.16:6379/0',
    backend='redis://10.240.187.16:6379/0',
)
