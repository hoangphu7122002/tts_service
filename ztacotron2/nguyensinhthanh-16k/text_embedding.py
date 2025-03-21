import random
import os
from ZaG2P.api import G2S

def cleaner(text, punctuations, eos):
  while text[-1] in [',', '.']:
      text = text[:-1]
  for p in punctuations:
      text = text.replace(p, ' ' + p + ' ')
  text = text.replace('. . .', ' . ').replace(' [', '[')
  text = text + ' ' + eos
  text = ' '.join(text.split()).strip().lower()
  return text

def word2phone(phone_dict_path, coda_nucleus_and_semivowel):
    word2phone_dict = {}
    if os.path.isfile(phone_dict_path):
        with open(phone_dict_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line and line.strip() and not line.startswith("#"):
                    items = line.strip().split(" ")
                    syllable = items[0]
                    if len(items) > 1:
                        curr_tone = items[1]
                        phonemes = items[2:]
                        result = []
                        for phoneme in phonemes:
                            if phoneme in coda_nucleus_and_semivowel:
                                result.append('@' + phoneme + curr_tone)
                            elif phoneme.isdigit():
                                curr_tone = phoneme
                            else:
                                result.append('@' + phoneme)
                        word2phone_dict[syllable] = result
                    else:
                        word2phone_dict[syllable] = {}
    return word2phone_dict

def phone2numeric(word2phone_dict):
    phone_lst = sorted(set([_phoneme for word in word2phone_dict.values() for _phoneme in word]))
    phone2numeric_dict = {phoneme: i for i, phoneme in enumerate(phone_lst)}
    return phone2numeric_dict

def symbol2numeric(hparams):
    letters_lst = list(hparams.letters)
    if hparams.p_phone_mix > 0:
        vn2phone_dict = word2phone(hparams.phone_vn_train, hparams.coda_nucleus_and_semivowel)
        oov2phone_dict = word2phone(hparams.phone_oov_train, hparams.coda_nucleus_and_semivowel)
        word2phone_dict = {**vn2phone_dict, **oov2phone_dict}
        phone2numeric_dict = phone2numeric(word2phone_dict)
        phonemes_lst = list(phone2numeric_dict.keys())
        if hparams.p_phone_mix < 1:
            symbols = letters_lst + phonemes_lst
        else:
            symbols = phonemes_lst
    else:
        symbols = letters_lst
    symbols = list(' ' + hparams.punctuation) + symbols
    if hparams.eos not in symbols:
        symbols = symbols + list(hparams.eos)
    symbol2numeric_dict = {s: i for i, s in enumerate(symbols)}
    return symbol2numeric_dict

def g2p(text_input, g2p_model, g2p_dict, word2phone_dict):
    text_output = ''
    words = [word for word in text_input.split(' ') if word]
    for word in words:
        if word not in word2phone_dict.keys():
            syllables = ' '.join([syl for syl in G2S(word, g2p_model, g2p_dict)[0].split(' ')[1:] if '(' not in syl])
            text_output += syllables + ' '
        else:
            text_output += word + ' '
    return text_output[:-1]

def text_to_sequence(text, p_phone_mix, word2phone_dict, symbol2numeric_dict):
    sequence = []
    words_of_text = [word for word in text.split(' ') if word]
    for word in words_of_text:
        if random.random() < p_phone_mix:
            if word in word2phone_dict.keys() and word2phone_dict[word]:
                phonemes = word2phone_dict[word]
                for phoneme in phonemes:
                    sequence.append(symbol2numeric_dict[phoneme])
            elif p_phone_mix >= 1:
                if word in symbol2numeric_dict.keys():
                    sequence.append(symbol2numeric_dict[word])
                else:
                    print(("{} not in phone_train_dict".format(word)))
        else:
            for symbol in word:
                if symbol in symbol2numeric_dict.keys():
                    sequence.append(symbol2numeric_dict[symbol])
                else:
                    print(("{} not in symbols_dict".format(symbol)))
                    print(text)
        sequence.append(symbol2numeric_dict[' '])
    return sequence[:-1]


if __name__ == '__main__':
    from hparams import create_hparams_and_paths
    hparams, path = create_hparams_and_paths()
    word2phone_dict = word2phone(hparams.phone_vn_train, hparams.coda_nucleus_and_semivowel)
    print(word2phone_dict)
    symbol2numeric_dict = symbol2numeric(hparams)
    print(symbol2numeric_dict)
    text = 'tôi là lâm he-llo'
    from ZaG2P.api import load_model
    #g2p_model, viet_dict = load_model()
    #text_out = g2p(text, g2p_model, viet_dict, word2phone_dict)
    #print(text_out)
    sequence = text_to_sequence(text, hparams.p_phone_mix, word2phone_dict, symbol2numeric_dict)
    print(sequence)
