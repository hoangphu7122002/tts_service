import os
import webrtcvad
from scipy.io.wavfile import read, write
from logmmse import logmmse
from vad import read_wave, write_wave, frame_generator, vad_collector
from hparams import create_hparams_and_paths
import librosa
import numpy as np
import argparse
from multiprocessing import Pool, cpu_count
from utils import progbar, stream

def valid_n_workers(num):
    n = int(num)
    if n < 1:
        raise argparse.ArgumentTypeError('%r must be an integer greater than 0' % num)
    return n

parser = argparse.ArgumentParser()
parser.add_argument('--hparams', '-p', type=str, help='comma separated name=value pairs')
parser.add_argument('--num_workers', '-n', metavar='N', type=valid_n_workers, default=int(cpu_count() * 0.8),
                    help='The number of worker threads')

args = parser.parse_args()

hp, paths = create_hparams_and_paths(args.hparams)
os.makedirs(paths.wavs_train_dir_path, exist_ok=True)
os.chmod(paths.wavs_train_dir_path, 0o775)

def process_wav(wav_name):
    wav_raw_path = f'{paths.wavs_raw_dir_path}/{wav_name}'
    wav_train_path = f'{paths.wavs_train_dir_path}/{wav_name}'
    wav_current_path = wav_raw_path
    sampling_rate, wav_train = read(wav_raw_path)
    if hp.norm_volume:
        p = "{'print $3'}"
        os.system(
            f'avg=`sox {wav_raw_path} -n stat 2>&1 | grep "Volume adjustment" | awk {p}` && coef=`echo "$avg * {hp.volume_ratio}" | bc` && sox -v $coef {wav_raw_path} {wav_train_path} > /dev/null 2>&1')
        wav_current_path = wav_train_path
    if sampling_rate != hp.sampling_rate:
        os.system(f"ffmpeg -y -i {wav_raw_path} -ar {hp.sampling_rate} {wav_train_path} > /dev/null 2>&1")
        wav_current_path = wav_train_path
    if hp.denoise:
        sampling_rate, wav_train = read(wav_current_path)
        wav_train = logmmse(wav_train, sampling_rate, initial_noise=hp.noise_frame)
        write(wav_train_path, sampling_rate, wav_train)
        wav_current_path = wav_train_path
    if hp.vad:
        wav_train, sampling_rate = read_wave(wav_current_path)
        vad = webrtcvad.Vad(hp.vad_aggressiveness)
        frames = frame_generator(30, wav_train, sampling_rate)
        frames = list(frames)
        segments = vad_collector(sampling_rate, 30, 300, vad, frames)
        data = bytearray([])
        for i, segment in enumerate(segments):
            data += segment
        write_wave(wav_train_path, data, sampling_rate)
        wav_current_path = wav_train_path
    if hp.trim_silence:
        sampling_rate, wav_train = read(wav_current_path)
        wav_train = wav_train / 32768.0
        wav_train, _ = librosa.effects.trim(wav_train, top_db=hp.trim_top_db)
        wav_train = wav_train * 32768.0
        wav_train = wav_train.astype(np.int16)
        write(wav_train_path, sampling_rate, wav_train)
    return True

pair_list = []
for wav_name in os.listdir(paths.wavs_raw_dir_path):
    pair_list.append(wav_name)

n_workers = max(1, args.num_workers)
print(f'num_workers = {args.num_workers}')
pool = Pool(processes=n_workers)
for i, status in enumerate(pool.imap_unordered(process_wav, pair_list), 1):
    bar = progbar(i, len(pair_list))
    message = f'{bar} {i}/{len(pair_list)}'
    stream(message)

print(f'\nCreated wavs_train at {paths.data_dir_path}')
