import os
import random
import argparse
import json
import torch
import torch.utils.data
import sys
from stft import TacotronSTFT
import numpy as np
from paths import Paths
from utils.file import files_to_list
from utils.audio import to_float, preprocess_wav, pre_emphasize, write_mel
import librosa


class Mel2Samp(torch.utils.data.Dataset):
    def __init__(self, files_list, trim_silence, trim_top_db, pre_emphasize, pre_emphasis, rescale, rescale_max, segment_length, sampling_rate, filter_length, hop_length, win_length, n_mel_channels, mel_fmin, mel_fmax):
        self.stft = TacotronSTFT(filter_length=filter_length,
                                 hop_length=hop_length,
                                 win_length=win_length,
                                 n_mel_channels = n_mel_channels,
                                 sampling_rate=sampling_rate,
                                 mel_fmin=mel_fmin, mel_fmax=mel_fmax)
        
        self.trim_silence = trim_silence
        self.trim_top_db = trim_top_db
        self.pre_emphasize = pre_emphasize
        self.pre_emphasis = pre_emphasis
        self.rescale = rescale
        self.rescale_max = rescale_max
        
        self.segment_length = segment_length
        self.sampling_rate = sampling_rate
        
        self.audio_files = files_to_list(files_list)
        random.seed(1234)
        random.shuffle(self.audio_files)
        

    def get_mel(self, audio):
        if type(audio) is not np.ndarray:
            audio = audio.squeeze().numpy()
        if self.pre_emphasize:            
            audio = pre_emphasize(audio, self.pre_emphasis) 
        melspec = self.stft.mel_spectrogram(audio)
        melspec = torch.from_numpy(melspec)
        return melspec

    def __getitem__(self, index):
        filename = self.audio_files[index]
        audio, sampling_rate = librosa.load(filename, sr=self.sampling_rate)
        audio = preprocess_wav(audio, self.sampling_rate, use_denoise=False, use_trim_silence=self.trim_silence, trim_top_db=self.trim_top_db, use_rescale=self.rescale, rescale_max=self.rescale_max)       
        audio = torch.FloatTensor(audio.astype(np.float32))
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start:audio_start+self.segment_length]
        else:
            print('Warning: {} is less than {} samples'.format(filename, self.segment_length))
            audio = torch.nn.functional.pad(audio, (0, self.segment_length - audio.size(0)), 'constant').data
            
        mel = self.get_mel(audio)    

        return (mel, audio)

    def __len__(self):
        return len(self.audio_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="config.json",
                        help='JSON file for configuration')
    parser.add_argument('-t', '--test', type=str, default='',
                        help='wav_file_path or test_list_file_path')
    parser.add_argument('-o', '--output_dir', type=str,
                        help='Output directory')
    args = parser.parse_args()
   
    with open(args.config) as f:
        lines = f.read()
    config = json.loads(lines)
    paths = Paths(config['data'], config['version'])
    if not args.test:
        args.test = paths.test_list_file_path
    mel2samp = Mel2Samp(args.test, **config["audio_config"], **config["feature_config"])   
    filepaths = files_to_list(args.test)
    if not args.output_dir:
        output_dir = os.path.join(config["data"], 'mel_spec')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        os.chmod(output_dir, 0o775)

    for filepath in filepaths:
        print(f"wav: {filepath}")
        audio, sampling_rate = librosa.load(filepath, config["feature_config"]["sampling_rate"])
        mel_filename = os.path.basename(filepath)
        mel_filepath = os.path.join(output_dir, mel_filename.replace('.wav', ''))         
        mel = mel2samp.get_mel(audio)
        mel = mel.squeeze().cpu().detach().numpy()    
        write_mel(mel, mel_filepath)
        print(f"mel: {mel_filepath}.npy\n")
       