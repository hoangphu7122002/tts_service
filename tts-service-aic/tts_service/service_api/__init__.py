#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @AUTHOR : thangdc94
# @Date : 2019/12/12
from flask import Flask
from flask_api import FlaskAPI

from tts_service.service_api.routes import *
from tts_service.service_api.routes import api_route, download_route


def create_app(config_dict) -> Flask:
    # create app instance
    app = Flask(__name__)
    # add configuration
    app.config.update(config_dict)
    # register blueprints
    app.register_blueprint(api_route.api)
    app.register_blueprint(download_route.api)
    return app
