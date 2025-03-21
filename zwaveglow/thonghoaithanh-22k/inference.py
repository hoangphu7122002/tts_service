import os
from scipy.io.wavfile import write
import torch
from mel2samp import files_to_list, MAX_WAV_VALUE, load_wav_to_torch, Mel2Samp
from glow import Denoiser
import time
import wave
import contextlib
import math
import json
import librosa
from tensorboardX import SummaryWriter   
from plotting import plot_spectrogram_to_numpy
import argparse


def main(test_file_path=None, sigma=0.8, denoiser_str=0.001, checkpoint_path='last'): 
    if test_file_path == None:
        test_file_path = os.path.join(data, 'test.txt')
    test_list = files_to_list(test_file_path)
    speaker = os.path.basename(data)
    speaker_dir = os.path.join('experiments', speaker)
    checkpoints_dir = os.path.join(speaker_dir, 'models', version)
    outputs_dir = os.path.join(speaker_dir, 'samples')
    logs_dir = os.path.join(speaker_dir, 'logs', version)    
    
    if not os.path.isdir(outputs_dir):
        os.makedirs(outputs_dir)     
        print("\noutputs directory:", outputs_dir)      
    if not os.path.isdir(logs_dir):
        os.makedirs(logs_dir)
        print("logs directory:", logs_dir)
        print()    
    logger = SummaryWriter(logs_dir)
    
    if checkpoint_path == 'last':     
        part1 = 'waveglow_' + version + '_'
        part3 = 'k'
        part2s = [int(chp.replace(part1, '').replace(part3, '')) for chp in os.listdir(checkpoints_dir) if chp.startswith(part1)] 
        checkpoint_path = os.path.join(checkpoints_dir, part1 + str(max(part2s)) + part3)      
    print('Loading model ' + checkpoint_path)
    
    checkpoint = torch.load(checkpoint_path)
    waveglow = checkpoint['model'] 
    iteration = checkpoint['iteration']
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow.cuda().eval()   
    
    mel2samp = Mel2Samp(test_file_path, **audio_config)
    if is_fp16:
        from apex import amp
        waveglow, _ = amp.initialize(waveglow, [], opt_level="O3") 
    if denoiser_str > 0:
        denoiser = Denoiser(waveglow).cuda() 
    print('Ready to synthesis \n')
    sampling_rate = audio_config["sampling_rate"]
    for i, audio_ori_path in enumerate(test_list):
        audio_ori_name = os.path.basename(audio_ori_path)
        audio_ori, sr = load_wav_to_torch(audio_ori_path, audio_config["trim_silence"]) 
        logger.add_audio("audio_ori_" + str(i+1), audio_ori, global_step=0, sample_rate=sr) 
        mel_ori = mel2samp.get_mel(audio_ori)       
        logger.add_image("mel_target_" + str(i+1), plot_spectrogram_to_numpy(mel_ori.data.cpu().numpy()), 0)
        mel_ori = torch.autograd.Variable(mel_ori.cuda())
        mel_ori = torch.unsqueeze(mel_ori, 0)
        mel_ori = mel_ori.half() if is_fp16 else mel_ori        
               
        with torch.no_grad():
            audio_out = waveglow.infer(mel_ori, sigma=sigma)   
            if denoiser_str > 0:
                audio_out = denoiser(audio_out, denoiser_str)                 
            logger.add_audio("audio_synthesys_" + str(i+1), audio_out, global_step=iteration, sample_rate=sampling_rate)      
            audio_out = audio_out * MAX_WAV_VALUE
        audio_out = audio_out.squeeze().cpu()                    
        audio_out = audio_out.numpy()
        audio_out_name = os.path.splitext(os.path.basename(audio_ori_path))[0]
        audio_out_name = os.path.basename(checkpoint_path) + '_' + str(sigma) + '_' + str(denoiser_str) + '_' + audio_out_name + '.wav'
        audio_out_path = os.path.join(
            outputs_dir, "{}".format(audio_out_name))  
        write(audio_out_path, sampling_rate, audio_out.astype('int16'))
        audio_out, sr = load_wav_to_torch(audio_out_path, audio_config["trim_silence"])
        mel_out = mel2samp.get_mel(audio_out)       
        logger.add_image("mel_predicted_" + str(i+1), plot_spectrogram_to_numpy(mel_out.data.cpu().numpy()), iteration)
        print(audio_out_name)     

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--cuda', type=str, default='0')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    config = 'config.json'
    with open(config) as f:
        lines = f.read()
    config = json.loads(lines)
    global data
    data = config["data"]
    global version
    version = config["version"]
    global audio_config
    audio_config = config["audio_config"] 
    global is_fp16
    is_fp16 = config["train_config"]["fp16_run"] 
    torch.backends.cudnn.benchmark = False
    main()