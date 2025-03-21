import torch
from torch_stft import STFT
from utils.audio import to_float, build_mel_basis, linear2mel, dynamic_range_compression
import numpy as np

class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0, mel_fmax=8000.0):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        self.mel_basis = build_mel_basis(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)

    def audio2torch_stft(self, audio):
        if type(audio) is np.ndarray:
            audio = to_float(audio)
            audio = torch.FloatTensor(audio.astype(np.float32))
            audio = audio.unsqueeze(0)
            audio = torch.autograd.Variable(audio, requires_grad=False)     
        return audio

    def mel_spectrogram(self, audio):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        audio: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)

        assert(torch.min(audio.data) >= -1)
        assert(torch.max(audio.data) <= 1)
        """
        audio = self.audio2torch_stft(audio)
        magnitudes, phases = self.stft_fn.transform(audio)
        magnitudes = magnitudes.data.squeeze().numpy()
        mel_output = linear2mel(magnitudes, self.mel_basis)
        mel_output = dynamic_range_compression(mel_output)
        return mel_output
