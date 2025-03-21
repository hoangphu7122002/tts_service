FROM python:3.6-slim-buster

WORKDIR /app

RUN apt-get update \
    && apt-get install -y ffmpeg sox

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

ENV APP_MAIN_FILE=tts_service.tts_main_service

COPY bin/ bin/
COPY tts_service/ tts_service/
COPY logging.yaml/ .

RUN mkdir logs/

RUN chmod +x bin/*

CMD ./bin/start.sh
