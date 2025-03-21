import os
import time


def call_api(voice, text):
    start = time.time()
    request_cmd = 'curl -X POST https://merlin-tts.kiki.laban.vn/api/end2end/path ' + '-F voice="{}"'.format(voice) + ' ' '-F text="{}"'.format(text)
    print(request_cmd)
    os.system(request_cmd)
    print(time.time() - start)


if __name__ == '__main__':
    voice = 'end2end_doanngocle'
    text = 'ông là người viết và đọc bản tuyên ngôn độc lập khai sinh nước việt nam dân chủ cộng hòa 1'
    call_api(voice, text)
