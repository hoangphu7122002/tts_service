#!/usr/bin/env bash

# Main class name, must be set
export APP_MAIN_FILE=tts_service.tts_main_service

# Daemonize
export APP_PID_DIR=.

export CONDA_ENV_NAME=tts

# Set run mode
if [ -z ${RUNMODE} ]; then
    export RUNMODE="dev"
fi
