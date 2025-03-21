#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @AUTHOR : thangdc94
# @Date : 2019/12/20
import os

from flask import Blueprint, send_from_directory

from tts_service import config
from tts_service.utils import logutils

api = Blueprint('download', __name__, url_prefix='/')

logger = logutils.getLogger(__name__)


@api.route('/data/<path:path>')
def send_data(path):
    return send_from_directory(config.SITE_ROOT, os.path.join(config.STORAGE_DATA, path))


@api.route('/chunk/<path:path>')
def send_chunk(path):
    return send_from_directory(config.SITE_ROOT, os.path.join(config.STORAGE_CHUNK, path))
