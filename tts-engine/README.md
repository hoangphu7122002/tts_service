# TTS engine

- Download VnCoreNLP Lib
```shell
git submodule update --init --recursive
```

- Start worker TTS
```shell
python worker_main.py --voice end2end_hoamai_rt --cuda 1 --new True
```

- Build and deploy using docker-compose

```shell
docker-compose build
docker-compose up -d
```