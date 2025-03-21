import numpy as np
from numpy import finfo
import torch
from model import Tacotron2
from text_embedding import text_to_sequence, word2phone, cleaner, g2p
from glow import Denoiser
from scipy.io.wavfile import write
from ZaG2P.api import load_model
import os
import time
from logmmse import logmmse


phone_oov_gen_dict_path = "dicts/phone_oov"
use_g2p = True

def load_tacotron_model(hparams):
    model = Tacotron2(hparams).cuda()
    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min
    return model

def prepare_models_and_dicts(voice='numiennam', phone_oov_gen_dict_path=phone_oov_gen_dict_path):
    print('Preparing models ...')
    print('Tacotron ...')

    if voice == 'doanngocle':
        from experiments.doanngocle.tacotron_models.v1.hparams_v1 import hparams
        tacotron2_model_path = 'experiments/doanngocle/tacotron_models/v1/tacotron_v1_80k'
        waveglow_model_path = 'waveglow_models/waveglow_doanngocle_v2'
        sigma = 0.666
        denoiser_strength = 0.01
        checkpoint_dict = torch.load(tacotron2_model_path)

    elif voice == 'camhieu':
        tacotron2_model_path = 'experiments/camhieu/tacotron_models/v5/tacotron2_202k.pt'
        waveglow_model_path = 'waveglow_models/waveglow_numiennam_v1'
        sigma = 0.666
        denoiser_strength = 0.01
        checkpoint_dict = torch.load(tacotron2_model_path)
        hparams = checkpoint_dict['hparams']

    else:
        raise ValueError(f'{voice} is not supported voice')

    tacotron_model = load_tacotron_model(hparams)
    tacotron_model.load_state_dict(checkpoint_dict['state_dict'])
    _ = tacotron_model.cuda().eval().half()

    print('Waveglow ...')
    waveglow_model = torch.load(waveglow_model_path)['model']
    waveglow_model.cuda().eval().half()
    for k in waveglow_model.convinv:
        k.float()

    print('Denoiser ...')
    denoiser_model = Denoiser(waveglow_model).cuda()

    print('G2P ...')
    g2p_model, g2p_dict = load_model()

    print('Preparing dictionaries...')
    vn2phone_train_dict = checkpoint_dict['vn2phone_train_dict']
    oov2phone_train_dict = checkpoint_dict['oov2phone_train_dict']
    oov2phone_gen_dict = word2phone(phone_oov_gen_dict_path, hparams.coda_nucleus_and_semivowel)
    if os.path.isfile(phone_oov_gen_dict_path):
        word2phone_dict = {**vn2phone_train_dict, **oov2phone_gen_dict}
    else:
        word2phone_dict = {**vn2phone_train_dict, **oov2phone_train_dict}
    symbol2numeric_dict = checkpoint_dict['symbol2numeric_dict']

    print('TTS is ready ...')
    return tacotron_model, waveglow_model, denoiser_model, g2p_model, g2p_dict, word2phone_dict, symbol2numeric_dict, hparams.p_phone_mix, hparams.punctuation, hparams.eos, \
           sigma, denoiser_strength, hparams.sampling_rate


def text2mel(text, tacotron_model, g2p_model, g2p_dict,  word2phone_dict, symbol2numeric_dict, p_phone_mix, punctuations, eos, use_g2p=use_g2p):
    text = cleaner(text, punctuations, eos)
    if use_g2p:
         text = g2p(text, g2p_model, g2p_dict, word2phone_dict)
    print(f'Text: {text}')
    sequence = np.array(text_to_sequence(text, p_phone_mix, word2phone_dict, symbol2numeric_dict))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
    mel_spec, mel_spec_postnet, _, alignments = tacotron_model.inference(sequence)
    return mel_spec_postnet


def text2speech(text, tacotron_model, waveglow_model, denoiser_model, g2p_model, g2p_dict, word2phone_dict, symbol2numeric_dict, p_phone_mix, punctuations, eos, use_g2p=use_g2p,
                sigma=0.666, denoiser_strength=0.01):
    s = time.time()
    mel_spec_postnet = text2mel(text, tacotron_model, g2p_model, g2p_dict, word2phone_dict, symbol2numeric_dict,  p_phone_mix, punctuations, eos, use_g2p)
    with torch.no_grad():
        wav = waveglow_model.infer(mel_spec_postnet, sigma=sigma)
        if denoiser_strength > 0:
            wav = denoiser_model(wav, strength=denoiser_strength)
    wav = wav.squeeze().cpu().numpy()
    #wav = logmmse(wav.astype(np.float64), 16000)
    speed = len(wav) / (time.time() - s)
    print(f'Speed: {int(speed // 1000)}kHz')
    return wav


def save_wav(wav_numpy_array, wav_save_path='test.wav', sample_rate=16000):
    data = wav_numpy_array * 32768.0
    data = data.astype('int16')
    write(wav_save_path, sample_rate, data)
    print(f'Output: {wav_save_path}')


def save_mel(mel_numpy_array, mel_save_path='test.mel'):
    np.save(mel_save_path, mel_numpy_array)
    print(f'Output: {mel_save_path}')