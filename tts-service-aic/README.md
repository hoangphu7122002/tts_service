# TTS Core API

- Start service
```shell
./bin/daemon.sh start
```

- Setup Flower to monitor task

```shell
docker run -d --rm --name flower-tts -v $(pwd)/tts_service/mycelery.py:/data/mycelery.py -p 5555:5555 mher/flower celery -A mycelery:app flower --purge_offline_workers=1
```

- Deploy using docker-compose

```shell
docker-compose up --build -d
```