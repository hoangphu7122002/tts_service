import torch
import numpy as np
import os
import csv
import sys
import math

def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).byte()
    return mask


def to_gpu(x):
    x = x.contiguous()
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)


def load_wav_name_and_text(metadata_path, delimiter='|'):
    extension = os.path.splitext(metadata_path)[1]
    audio_name_and_text = []
    with open(metadata_path, encoding='utf-8') as f:
        if extension == '.csv':
            csv_reader = csv.reader(f, delimiter=',')
            for row in csv_reader:
                audio_name = row[0]
                if '.wav' not in audio_name:
                    audio_name = audio_name + '.wav'
                text = row[1]
                audio_name_and_text.append([audio_name, text])
        else:
            splits = [line.strip().split(delimiter, 1) for line in f if line]
            for pair in splits:
                audio_name = pair[0]
                if '.wav' not in audio_name:
                    audio_name = audio_name + '.wav'
                text = pair[1]
                audio_name_and_text.append([audio_name, text])
    return audio_name_and_text


def to_float(_input):
    if _input.dtype == np.float64:
        return _input, _input.dtype
    elif _input.dtype == np.float32:
        return _input.astype(np.float64), _input.dtype
    elif _input.dtype == np.uint8:
        return (_input - 128) / 128., _input.dtype
    elif _input.dtype == np.int16:
        return _input / 32768., _input.dtype
    elif _input.dtype == np.int32:
        return _input / 2147483648., _input.dtype
    raise ValueError('Unsupported wave file format {}'.format(_input.dtype))


def from_float(_input, dtype):
    if dtype == np.float64:
        return _input, np.float64
    elif dtype == np.float32:
        return _input.astype(np.float32)
    elif dtype == np.uint8:
        return ((_input * 128) + 128).astype(np.uint8)
    elif dtype == np.int16:
        return (_input * 32768).astype(np.int16)
    elif dtype == np.int32:
        print(_input)
        return (_input * 2147483648).astype(np.int32)
    raise ValueError('Unsupported wave file format'.format(_input.dtype))


def progbar(k, n, size=32):
    done = (k * size) // n
    bar = ''
    for i in range(size):
        bar += '█' if i <= done else '░'

    return bar


def stream(message):
    sys.stdout.write(f"\r{message}")


def simple_table(item_tuples):
    print()

    border_pattern = '+---------------------------------------'
    whitespace = '                                            '

    headings, cells, = [], []

    for item in item_tuples:

        heading, cell = str(item[0]), str(item[1])

        pad_head = True if len(heading) < len(cell) else False

        pad = abs(len(heading) - len(cell))
        pad = whitespace[:pad]

        pad_left = pad[:len(pad)//2]
        pad_right = pad[len(pad)//2:]

        if pad_head:
            heading = pad_left + heading + pad_right
        else:
            cell = pad_left + cell + pad_right

        headings += [heading]
        cells += [cell]

    border, head, body = '', '', ''

    for i in range(len(item_tuples)):

        temp_head = f'| {headings[i]} '
        temp_body = f'| {cells[i]} '

        border += border_pattern[:len(temp_head)]
        head += temp_head
        body += temp_body

        if i == len(item_tuples) - 1:
            head += '|'
            body += '|'
            border += '+'

    print(border)
    print(head)
    print(border)
    print(body)
    print(border)
    print(' ')
