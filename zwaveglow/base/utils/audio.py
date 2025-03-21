import librosa
import numpy as np
from scipy.io.wavfile import write
from scipy import signal
from logmmse import logmmse
from librosa.filters import mel
from torch_stft import STFT
import torch


def to_float(_input):
    if _input.dtype == np.float64:
        return _input
    elif _input.dtype == np.float32:
        return _input.astype(np.float64)
    elif _input.dtype == np.uint8:
        return (_input - 128) / 128.
    elif _input.dtype == np.int16:
        return _input / 32768.
    elif _input.dtype == np.int32:
        return _input / 2147483648.
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
        return (_input * 2147483648).astype(np.int32)
    raise ValueError('Unsupported wave file format'.format(_input.dtype))

def trim_silence(wav, top_db):
    wav = to_float(wav)
    wav = librosa.effects.trim(wav, top_db=top_db)[0]
    return wav

def denoise(wav, sampling_rate, noise_frame):
    wav = logmmse(wav, sampling_rate, initial_noise=noise_frame)
    return wav

def pre_emphasize(wav, pre_emphasis):
    wav = to_float(wav)
    wav = signal.lfilter([1, -pre_emphasis], [1], wav)
    return wav

def rescale(wav, rescale_max):
    wav = wav / max(0.01, np.abs(wav).max()) * rescale_max
    return wav

def write_wav(wav, wav_save_path='test.wav', sampling_rate=16000, use_rescale=False, rescale_max=0.999):
    if use_rescale:
        wav = rescale(wav, rescale_max)
    wav = from_float(wav, np.int16)
    write(wav_save_path, sampling_rate, wav)

def write_mel(mel, mel_save_path='test.mel'):
    np.save(mel_save_path, mel)

def dynamic_range_compression(x, clip_val=1e-5, C=1):
    return np.log(np.maximum(clip_val, x) * C)

def build_mel_basis(sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax):
    mel_basis = mel(sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
    return mel_basis

def stft_torch(wav, filter_length, hop_length, win_length):
    if type(wav) is np.ndarray:
        wav = to_float(wav)
        wav = torch.FloatTensor(wav.astype(np.float32))
        audio = wav.unsqueeze(0)
        wav = torch.autograd.Variable(audio, requires_grad=False)
    stft_fn = STFT(filter_length, hop_length, win_length)
    magnitudes, phases = stft_fn.transform(wav)
    magnitude = magnitudes.squeeze().numpy()
    return magnitude

def stft_librosa(wav, filter_length, hop_length, win_length):
    spectrogram = librosa.stft(y=wav, n_fft=filter_length, hop_length=hop_length, win_length=win_length,
                        pad_mode='constant')
    return spectrogram

def linear2mel(spectrogram, mel_basis):
    melspectrogram = np.dot(mel_basis, spectrogram)
    return melspectrogram

def melspectrogram_torch(wav, mel_basis, filter_length, hop_length, win_length, magnitude_power=1, min_level_db=-100, norm_mel=False, symmetric_norm=True, max_abs_value=1):
    magnitude = stft_torch(wav, filter_length, hop_length, win_length)
    magnitude = magnitude ** magnitude_power
    melspectrogram = linear2mel(magnitude, mel_basis)
    melspectrogram = dynamic_range_compression(melspectrogram, min_level_db)
    if norm_mel:
        melspectrogram = normalize(melspectrogram, symmetric_norm, max_abs_value, min_level_db)
    return melspectrogram

def melspectrogram_librosa(wav, mel_basis, filter_length, hop_length, win_length, magnitude_power=2, min_level_db=-100, ref_level_db=20, norm_mel=True, symmetric_norm=True, max_abs_value=1):
    spectrogram = stft_librosa(wav, filter_length, hop_length, win_length)
    magnitude = np.abs(spectrogram) ** magnitude_power
    melspectrogram = linear2mel(magnitude, mel_basis)
    melspectrogram = 20 * dynamic_range_compression(melspectrogram, min_level_db) - ref_level_db
    if norm_mel:
        melspectrogram = normalize(melspectrogram, symmetric_norm, max_abs_value, min_level_db)
    return melspectrogram

def normalize(x, symmetric_norm=True, max_abs_value=1, min_level_db=-100):
    if symmetric_norm:
        return np.clip((2 * max_abs_value) * ((x - min_level_db) / (-min_level_db)) - max_abs_value, -max_abs_value, max_abs_value)
    else:
        return np.clip(max_abs_value * ((x - min_level_db) / (-min_level_db)), 0, max_abs_value)

def preprocess_wav(audio, sampling_rate, use_denoise, noise_frame=6, use_trim_silence=True, trim_top_db=40, use_rescale=True, rescale_max=0.8):
    if use_denoise:
        audio = denoise(audio, sampling_rate, noise_frame)
    if use_trim_silence:
        audio = trim_silence(audio, trim_top_db)
    if use_rescale:
        audio = rescale(audio, rescale_max)
    return audio
