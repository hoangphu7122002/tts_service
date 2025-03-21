from tts_service import config
from tts_service.service_api import create_app
from flask_apscheduler import APScheduler
import os
from itertools import chain
import time

def interval_task():
    print("Run cleaning task")
    now = time.time()
    all_data_dirs = [config.STORAGE_DATA, config.STORAGE_CHUNK]
    for root, dirs, files in chain.from_iterable(os.walk(data_dir) for data_dir in all_data_dirs):
        for file in files:
            full_path = os.path.join(root, file)
            if os.stat(full_path).st_mtime < now - 7 * 86400:
                os.remove(full_path)


def main():
    app = create_app(config.FLASK_SERVER_CONFIG)
    scheduler = APScheduler()
    scheduler.init_app(app)
    scheduler.start()
    scheduler.add_job(id="DELETE_DATA_DIR", func=interval_task, trigger='interval', days=1)
    app.run(host=config.FLASK_SERVER_CONFIG['HOST'],
            port=config.FLASK_SERVER_CONFIG['PORT'])


if __name__ == '__main__':
    main()
