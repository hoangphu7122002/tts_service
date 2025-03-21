import os


def word2phone(phone_dict_path, coda_nucleus_and_semivowel):
    word2phone_dict = {}
    with open(phone_dict_path, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
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


def word2phone_2(vn_dict_path, oov_dict_path, coda_nucleus_and_semivowel):
    vn2phone_dict = word2phone(vn_dict_path, coda_nucleus_and_semivowel)
    oov2phone_gen_dict = {}
    if os.path.isfile(oov_dict_path):
        oov2phone_gen_dict = word2phone(oov_dict_path, coda_nucleus_and_semivowel)
        print(oov2phone_gen_dict)
    word2phone_dict = {**vn2phone_dict, **oov2phone_gen_dict}
    return word2phone_dict


def phone2numeric(word2phone_dict):
    phone_lst = sorted(set([_phoneme for word in word2phone_dict.values() for _phoneme in word]))
    phone2numeric_dict = {phoneme: i for i, phoneme in enumerate(phone_lst)}
    return phone2numeric_dict


def symbols2numeric(hparams):
    letters_lst = list(hparams.letters)
    if hparams.p_phone_mix > 0:
        word2phone_dict = word2phone_2(hparams.phone_vn, hparams.phone_oov_train,
                                       hparams.coda_nucleus_and_semivowel)
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
