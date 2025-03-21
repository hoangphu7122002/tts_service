#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @AUTHOR : thangdc94
# @Date : 2019/12/20
from flask import Blueprint, jsonify, request, make_response
from flask_api import status

from tts_service import config
from tts_service.constants import return_codes
from tts_service.models.base_response import BaseResponse
from tts_service.tts_model import VoiceEnd2End
from tts_service.utils import logutils, jsonutils

api = Blueprint('api', __name__, url_prefix='/api')

logger = logutils.getLogger(__name__)

SPORT_DOMAIN = 1
MATH_DOMAIN = 2
NOT_SPECIFIED_DOMAIN = 3

voice_dict = {}
voice_infos = []
for voice_config in config.VOICE_CONFIGS:
    voice_infos.append(voice_config["info"])
    voice_dict[voice_config["id"]] = VoiceEnd2End(voice_config["storage_dir"],
                                                  request_queue=voice_config["request_queue_name"],
                                                  k_timeout=voice_config["k_timeout"],
                                                  k_abnormal=voice_config["k_abnormal"])


@api.after_request
def after_request_func(data):
    response = make_response(data)
    response.headers['Content-Type'] = 'application/json'
    return response


@api.route("/v1/voices", methods=["GET"])
def api_v1_voices():
    return jsonify(
        status=return_codes.API_SUCCESS,
        msg="Success!",
        voices=voice_infos,
        version=config.API_VERSION)


# @app.route("/api/v1/streaming", methods=["POST"])
def api_v1_streaming():
    if "text" in request.form:
        text = request.form["text"]
    else:
        return jsonify(
            status=return_codes.API_MISSING_FIELD,
            msg="Missing text field!")
    if "voiceId" in request.form:
        voice_id = request.form.get("voiceId", type=int)

    else:
        return jsonify(
            status=return_codes.API_MISSING_FIELD,
            msg="Missing voiceId field!")
    if voice_id in voice_dict:
        text_norm, audio_path, m3u8_path = voice_dict[voice_id].stream(NOT_SPECIFIED_DOMAIN, text, caching=False,
                                                                       normOOV=True)
    else:
        return jsonify(
            status=return_codes.API_ERROR,
            msg="Voice ID not existed")
    if m3u8_path is not None:
        return jsonify(
            status=return_codes.API_SUCCESS,
            msg="Success!",
            data={
                "url": "{}/{}".format(config.DOMAIN, m3u8_path),
                "retry": 40,
                "timeoutMilSecs": 100
            })
    return jsonify(
        status=return_codes.API_ERROR,
        msg="Something wrong!")


@api.route("/v1/path", methods=["POST"])
def api_v1_path():
    if "text" in request.form:
        text = request.form["text"]
    else:
        return jsonutils.to_json(BaseResponse(status=return_codes.API_MISSING_FIELD,
                                              msg="Missing text field!")), status.HTTP_400_BAD_REQUEST
    if "voiceId" in request.form:
        voice_id = request.form.get("voiceId", type=int)
    else:
        return jsonutils.to_json(BaseResponse(status=return_codes.API_MISSING_FIELD,
                                              msg="Missing voiceId field!")), status.HTTP_400_BAD_REQUEST
    if voice_id in voice_dict:
        return voice_dict[voice_id].speak_rabbitMQ(NOT_SPECIFIED_DOMAIN, text, caching=False, normOOV=True)
    else:
        return jsonutils.to_json(BaseResponse(status=return_codes.API_ERROR,
                                              msg="Voice ID not existed!")), status.HTTP_400_BAD_REQUEST
