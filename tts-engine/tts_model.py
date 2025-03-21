import logging
import sys

sys.path.append('models/')

from config import HPARAMS, SIGMA_WAVEGLOW, DENOISER_STRENGTH, WAVEGLOW_PATHS, TACOTRON_PATHS
import torch
import numpy as np
from numpy import finfo
from tacotron2 import Tacotron2
from glow import Denoiser
from text_embedding import TextEmbedding

logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s', level=logging.INFO)


class TtsDecodeError(Exception):
    pass


class VoiceEnd2End:
    def __init__(self, voice, new=False, use_eos=False):
        if new:
            print('New version')
        self.voice = voice
        self.new = new
        self.use_eos = use_eos
        if not self.new:
            self.hparams = HPARAMS[self.voice]
            self.sampling_rate = self.hparams.sampling_rate
        self.prepare_models()

    def prepare_models(self):
        logging.info('Preparing models ...')
        self.load_tacotron()
        self.load_waveglow()
        self.load_denoiser()
        self.load_text_embedding()
        logging.info('TTS is ready ...')

    def get_mel_spec_tacotron(self, text):
        sequence = np.array(self.text_embedding.text_to_sequence(text))[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
        mel_spec, mel_spec_postnet, _, alignments = self.tacotron_model.inference(sequence)
        return mel_spec_postnet

    def speech(self, text, use_g2p=True):
        text = self.text_embedding.cleaner(text, use_eos=False)
        if use_g2p:
            text = self.text_embedding.g2p(text)
        if self.use_eos:
            text = text + " " + self.hparams.eos
        logging.info("Speech: {}".format(text))
        mel_spec_postnet = self.get_mel_spec_tacotron(text)

        # Reached max decoder steps
        if mel_spec_postnet.shape[2] == self.hparams.max_decoder_steps:
            raise TtsDecodeError("Warning! Reached max decoder steps")
        with torch.no_grad():
            audio = self.waveglow_model.infer(mel_spec_postnet, sigma=SIGMA_WAVEGLOW)
            audio = self.denoiser_model(audio, strength=DENOISER_STRENGTH)
        audio = audio.squeeze().cpu().numpy()
        return audio

    def load_tacotron(self):
        tacotron_path = TACOTRON_PATHS[self.voice]
        logging.info('Tacotron (load from {}) ...'.format(tacotron_path))
        checkpoint_dict = torch.load(tacotron_path)
        if self.new:
            self.hparams = checkpoint_dict['hparams']
            self.sampling_rate = self.hparams.sampling_rate
            self.hparams.phone_vn = 'trained/dicts/phone_vn_south'
        self.tacotron_model = Tacotron2(self.hparams).cuda()
        if self.hparams.fp16_run:
            self.tacotron_model.decoder.attention_layer.score_mask_value = finfo('float16').min
        self.tacotron_model.load_state_dict(checkpoint_dict['state_dict'])
        _ = self.tacotron_model.cuda().eval().half()
        logging.info('Sample rate {} ...'.format(self.sampling_rate))

    def load_waveglow(self):
        waveglow_path = WAVEGLOW_PATHS[self.voice]
        logging.info('Waveglow (load from {}) ...'.format(waveglow_path))
        self.waveglow_model = torch.load(waveglow_path)['model']
        self.waveglow_model.cuda().eval().half()
        for k in self.waveglow_model.convinv:
            k.float()

    def load_denoiser(self):
        logging.info('Denoiser ...')
        self.denoiser_model = Denoiser(self.waveglow_model).cuda()

    def load_text_embedding(self):
        logging.info('TextEmbedding ...')
        self.text_embedding = TextEmbedding(self.voice, self.hparams)
