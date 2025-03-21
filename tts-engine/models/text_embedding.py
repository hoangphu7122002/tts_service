import sys

sys.path.append('../')
import random
from ZaG2P.api import G2S
from ZaG2P.api import load_model as g2p_load_model
from utils.text_utils import *
from config import READABLE_OOV_DICT_PATH


class TextEmbedding:
    def __init__(self, voice, hparams):
        """
        - g2p_model
        - g2p_dict
        - word2phone_dict
        - symbol2numeric_dict
        - p_phone_mix
        - punctuations
        - eos
        """
        print('G2P ...')
        self.g2p_model, self.g2p_dict, self.phoneme_syllable_vndict = g2p_load_model()
        self.p_phone_mix = hparams.p_phone_mix
        self.word2phone_dict = word2phone_2(hparams.phone_vn, READABLE_OOV_DICT_PATH,
                                            hparams.coda_nucleus_and_semivowel)
        self.symbol2numeric_dict = symbols2numeric(hparams)

        self.punctuations = hparams.punctuation
        self.eos = hparams.eos
        self.oov_g2p_file = open('oov_g2p_file.txt', 'a')

    def cleaner(self, text, use_eos=True):
        while text[-1] in self.punctuations:
            text = text[:-1]
        for p in self.punctuations:
            text = text.replace(p, ' ' + p + ' ')
        text = text.replace('. . .', ' . ')
        if use_eos:
            text = text + ' ' + self.eos
        text = ' '.join(text.split()).strip().lower()
        return text

    def text_to_sequence(self, text):
        sequence = []
        words_of_text = [word for word in text.split(' ') if word]
        for word in words_of_text:
            if random.random() < self.p_phone_mix:
                if word in self.word2phone_dict.keys() and self.word2phone_dict[word]:
                    phonemes = self.word2phone_dict[word]
                    for phoneme in phonemes:
                        sequence.append(self.symbol2numeric_dict[phoneme])
                elif self.p_phone_mix >= 1:
                    if word in self.symbol2numeric_dict.keys():
                        sequence.append(self.symbol2numeric_dict[word])
                    else:
                        print(("{} not in phone_train_dict".format(word)))
            else:
                for symbol in word:
                    if symbol in self.symbol2numeric_dict.keys():
                        sequence.append(self.symbol2numeric_dict[symbol])
                    else:
                        print(("{} not in symbols_dict".format(symbol)))
            sequence.append(self.symbol2numeric_dict[' '])
        return sequence[:-1]

    def g2p(self, text):
        text_output = ''
        words = [word for word in text.split(' ') if word]
        for word in words:
            if word not in self.word2phone_dict.keys():
                g2s_syllables = G2S(word, self.g2p_model, self.g2p_dict, self.phoneme_syllable_vndict)[0].split(' ')[1:]
                filtered_syllables = []
                for syl in g2s_syllables:
                    if syl is g2s_syllables[-1]:
                        if '(' not in syl:
                            filtered_syllables.append(syl)
                    else:
                        filtered_syllables.append(syl.replace("(", "").replace(")", ""))

                syllables = ' '.join(filtered_syllables)
                if syllables:
                    text_output += syllables + ' '
                else:
                    self.oov_g2p_file.writelines(word + "\n")
                    self.oov_g2p_file.flush()
                    text_output += " , "
            else:
                text_output += word + ' '
        return text_output[:-1]
