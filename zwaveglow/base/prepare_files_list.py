from utils.audio import trim_silence
from paths import Paths
import json
import argparse
import os
from scipy.io.wavfile import read

total_test_files = 2

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='config.json',
                        help='JSON file for configuration')
args = parser.parse_args()

with open(args.config) as f:
     lines = f.read()
config = json.loads(lines)
paths = Paths(config["data"], config["version"])

count_files = 0
all_list_file = open(paths.list_file_path, 'w')

count_train_files = 0
train_list_file = open(paths.train_list_file_path, 'w')

count_test_files = 0
test_list_file = open(paths.test_list_file_path, 'w')

for wav_name in os.listdir(paths.wavs_dir_path):
    wav_path = f"{paths.wavs_dir_path}/{wav_name}"
    all_list_file.write(wav_path + '\n')
    count_files += 1 
    sampling_rate, audio = read(wav_path)   
    if config["audio_config"]["trim_silence"]:
        audio = trim_silence(audio, config["audio_config"]["trim_top_db"])
    n_samples = len(audio) * config["feature_config"]["sampling_rate"] / sampling_rate        
    if count_test_files < total_test_files :
        test_list_file.write(wav_path + '\n')
        count_test_files += 1  
    else:        
        if n_samples > config["feature_config"]["segment_length"]:
            train_list_file.write(wav_path + '\n')
            count_train_files +=1
           
print(f"Created all.txt \t Total wavs files: {count_files}")
print(f"Created train.txt \t Total train wavs files: {count_train_files}")
print(f"Created test.txt \t Total test wav files: {count_test_files}")
                 