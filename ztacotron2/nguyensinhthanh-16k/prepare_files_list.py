import os
from hparams import create_hparams_and_paths
from text_embedding import cleaner
from math import inf
from utils import load_wav_name_and_text
from scipy.io.wavfile import read
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--hparams', '-p', type=str, help='comma separated name=value pairs')
args = parser.parse_args()

hp, paths = create_hparams_and_paths(args.hparams)

os.makedirs(paths.files_lists_dir_path, exist_ok=True)
os.chmod(paths.files_lists_dir_path, 0o775)

remove_files_list = [os.path.join(paths.data_dir_path, f) for f in os.listdir(paths.data_dir_path) if f.startswith('removed')]

wav_name_and_text = load_wav_name_and_text(paths.metadata_file_path, delimiter='|')
train_list = open(paths.train_list_file_path, 'w')
val_list = open(paths.val_list_file_path, 'w')
total_val_files = len(wav_name_and_text) // 100
count = 0
invalid_wavs = []
if hp.filter_audios:
    for remove_file_path in remove_files_list:
        if 'removed_oov' not in remove_file_path:
            with open(remove_file_path) as f:
                invalid_wavs += [line.replace('\n', '').split('|')[0] for line in f]
if hp.p_phone_mix >= 1:
    for remove_file_path in remove_files_list:
        if 'removed_oov' in remove_file_path:
            with open(remove_file_path) as f:
                invalid_wavs += [line.replace('\n', '').split('|')[0] for line in f]
training_duration = 0
valid_duration = 0
removed_duration = 0
for pair in wav_name_and_text:
    wav_name = pair[0]
    wav_path = os.path.join(paths.wavs_train_dir_path, wav_name)
    text = pair[1]
    text = cleaner(text, hp.punctuation, hp.eos)
    sr, data = read(wav_path)
    frames = len(data)
    duration = frames / sr
    if hp.longest_wav_in_seconds:
        longest_wav_in_seconds = float(hp.longest_wav_in_seconds)
    else:
        longest_wav_in_seconds = inf
    if wav_name not in invalid_wavs and duration <= longest_wav_in_seconds:
        if count < total_val_files:
            val_list.write(wav_path + '|' + text + '\n')
            valid_duration += duration / 3600
            count += 1
        else:
            train_list.write(wav_path + '|' + text + '\n')
            training_duration += duration / 3600
    else:
        removed_duration += duration / 36000

print(f'Created metadata_train.txt and metadata_val.txt at {paths.files_lists_dir_path}/')
print('Total training: {}h'.format(round(training_duration, 3)))
print('Total valid: {}h'.format(round(valid_duration, 3)))
print('Total removed: {}h'.format(round(removed_duration, 3)))
