import os
from scipy.io.wavfile import write
import torch
from mel2samp import Mel2Samp
from utils.file import files_to_list
from utils.display import simple_table
from utils.audio import write_wav, to_float
from glow import Denoiser
import wave
import contextlib
import math
import json
import librosa
from tensorboardX import SummaryWriter   
from plotting import plot_spectrogram_to_numpy
from paths import Paths
import argparse
import time
from scipy.io.wavfile import read
import librosa
    
    
def main(config_file_path, sigma, denoiser_str, checkpoint_path, test_file_path): 
    simple_table([('config_file', config_file_path),
                  ('sigma', sigma),
                 ('denoiser_strength', denoiser_str)])
    with open(config_file_path) as f:
        lines = f.read()
    config = json.loads(lines)
    paths = Paths(config["data"], config["version"])
    if not test_file_path:
        test_file_path = paths.test_list_file_path    
    test_list = files_to_list(test_file_path)
                   
    os.makedirs(paths.samples_dir_path, exist_ok=True)
    os.chmod(paths.samples_dir_path, 0o775)
    os.makedirs(paths.logs_dir_path, exist_ok=True)
    os.chmod(paths.logs_dir_path, 0o775)
    logger = SummaryWriter(paths.logs_dir_path)
    
    if checkpoint_path == 'last':     
        part1 = 'waveglow_' + os.path.basename(config["data"]) + '_' + config["version"] + '_'
        part3 = 'k.pt'
        part2s = [int(chp.replace(part1, '').replace(part3, '')) for chp in os.listdir(paths.checkpoints_dir_path) if chp.startswith(part1)] 
        checkpoint_path = os.path.join(paths.checkpoints_dir_path, part1 + str(max(part2s)) + part3)      
    print(f'Loading model from checkpoint {checkpoint_path}\n')
    
    checkpoint = torch.load(checkpoint_path)
    waveglow = checkpoint['model'] 
    iteration = checkpoint['iteration']
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow.cuda().eval()   
    
    mel2samp = Mel2Samp(test_file_path, **config["audio_config"], **config["feature_config"])
    is_fp16 = config["training_config"]["fp16_run"]
    if is_fp16:        
        from apex import amp
        waveglow, _ = amp.initialize(waveglow, [], opt_level="O3") 
    if denoiser_str > 0:
        denoiser = Denoiser(waveglow).cuda() 
           
    print(f"\ntest file path : {test_file_path}")
    print('\nReady to synthesis\n')
    
    sampling_rate = config["feature_config"]["sampling_rate"]
    for i, audio_ori_path in enumerate(test_list):
        print(f"ori wav: {audio_ori_path}")
        audio_ori_name = os.path.basename(audio_ori_path)
        audio_ori, sampling_rate = librosa.load(audio_ori_path, sr=sampling_rate) 
        logger.add_audio("audio_ori_" + str(i+1), to_float(audio_ori), global_step=0, sample_rate=sampling_rate) 
        mel_ori = mel2samp.get_mel(audio_ori)   
        logger.add_image("mel_target_" + str(i+1), plot_spectrogram_to_numpy(mel_ori.squeeze().cpu().numpy()), 0)
        mel_ori = torch.autograd.Variable(mel_ori.cuda())
        mel_ori = torch.unsqueeze(mel_ori, 0)
        mel_ori = mel_ori.half() if is_fp16 else mel_ori        
        s = time.time()  
        with torch.no_grad():
            audio_out = waveglow.infer(mel_ori, sigma=sigma)   
            if denoiser_str > 0:
                audio_out = denoiser(audio_out, denoiser_str) 
            decode_time = round(time.time() - s, 2)          
            logger.add_audio("audio_synthesys_" + str(i+1), audio_out, global_step=iteration, sample_rate=sampling_rate)      
        audio_out = audio_out.squeeze().cpu().numpy()                 
        print(f"time: {decode_time}")
        decode_speed = int(len(audio_out) / decode_time / 1000)
        print(f'speed: {decode_speed}kHz')
        audio_out_name = os.path.splitext(os.path.basename(audio_ori_path))[0]
        audio_out_name = os.path.basename(checkpoint_path).replace('.pt', '') + '_' + audio_out_name + '.wav'
        audio_out_path = f"{paths.samples_dir_path}/{audio_out_name}"  
        write_wav(audio_out, audio_out_path, sampling_rate)
        sampling_rate, audio_out = read(audio_out_path)
        mel_out = mel2samp.get_mel(audio_out)
        logger.add_image("mel_predicted_" + str(i+1), plot_spectrogram_to_numpy(mel_out.squeeze().cpu().numpy()), iteration)
        print(f"out wav: {audio_out_path}\n")     

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.json',
                        help='JSON file for configuration')
    parser.add_argument('-t', '--test', type=str, default='',
                        help='wav_file_path or test_list_file_path')
    parser.add_argument('-p', '--checkpoint', type=str, default='last')
    parser.add_argument('--sigma', type=float, default=0.8)
    parser.add_argument('--denoise', type=float, default=0.001)
    parser.add_argument('-d', '--cuda', type=str, default='0')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    torch.backends.cudnn.benchmark = False
    main(args.config, args.sigma, args.denoise, args.checkpoint, args.test)