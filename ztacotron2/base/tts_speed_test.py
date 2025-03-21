import numpy as np
import torch
from text_embedding import text_to_sequence, cleaner, g2p
import time
import threading
import librosa
from tts_infer import prepare_models_and_dicts, save_wav
import os


def tts(text, tacotron, waveglow, denoiser, g2p_model, viet_dict, word2phone_dict, symbol2numeric_dict, p_phone_mix, punctuations, eos, sigma, denoiser_strength, sampling_rate):
    # text to sequence
    t = time.time()
    text = cleaner(text, punctuations, eos)
    text = g2p(text, g2p_model, viet_dict, word2phone_dict)
    sequence = np.array(text_to_sequence(text, p_phone_mix, word2phone_dict, symbol2numeric_dict))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
    t_embedding = time.time() - t
    # sequence to mel-spec by Tacotron2
    t = time.time()
    mel_outputs, mel_outputs_postnet, _, alignments = tacotron.inference(sequence)
    t_tacotron = time.time() - t
    # mel-spec to audio by Waveglow
    t = time.time()
    with torch.no_grad():
        audio = waveglow.infer(mel_outputs_postnet, sigma=sigma)
        t_waveglow = time.time() - t
        t = time.time()
        audio = denoiser(audio, strength=denoiser_strength)
        t_denoiser = time.time() - t
        t = time.time()
    audio = audio.squeeze().cpu().numpy()
    audio_path = 'test.wav'
    save_wav(audio, audio_path, sampling_rate)
    t_write = time.time() - t
    y, sr = librosa.load(audio_path)
    duration = librosa.get_duration(y)
    return t_embedding, t_tacotron, t_waveglow, t_denoiser, t_write, duration


def speed_test(text, tacotron, waveglow, denoiser, g2p, viet_dict, word2phone_dict, symbol2numeric_dict, p_phone_mix, punctuation, eos, sigma, denoiser_strength, sampling_rate, times):
    time_e = time_t = time_w = time_d = time_wr = total_duration = 0
    for i in range(times):
        e, t, w, d, wr, duration = tts(text, tacotron, waveglow, denoiser, g2p, viet_dict, word2phone_dict, symbol2numeric_dict, p_phone_mix, punctuation, eos,sigma, denoiser_strength, sampling_rate)
        time_e += e
        time_t += t
        time_w += w
        time_d += d
        time_wr += wr
        total_duration += duration
    elapsed = time_e + time_t + time_w + time_d + time_w
    print("avg_total: {:.4f}s - avg_embedding: {:.4f}s - avg_tacotron: {:.4f}s - avg_waveglow: {:.4f}s - avg_denoiser: {:.4f}s - avg_write: {:.4f}s".format(
        elapsed / times, time_e / times, time_t / times, time_w / times, time_d / times, time_wr / times))
    print('avg_duration: {:.4f}'.format(total_duration / times))


if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "5"
    tacotron, waveglow, denoiser, g2p_model, viet_dict, word2phone_dict, symbol2numeric_dict, p_phone_mix, punctuations, eos, sigma, denoiser_strength,\
    sampling_rate = prepare_models_and_dicts(voice='camhieu')
    text = "tôi là lâm hello"
    print("\nStart ...\n")
    text_list = [text] * 1
    for text in text_list:
        threading.Thread(target=speed_test, args=(text, tacotron, waveglow, denoiser, g2p_model, viet_dict, word2phone_dict,
                                                  symbol2numeric_dict, p_phone_mix, punctuations, eos, sigma, denoiser_strength, sampling_rate, 2)).start()





