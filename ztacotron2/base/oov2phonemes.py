import re
import os
from hparams import create_hparams_and_paths
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--hparams', type=str, help='comma separated name=value pairs')
args = parser.parse_args()

hp, paths = create_hparams_and_paths(args.hparams)

oov2syllables_train_path = os.path.join(hp.data, 'syllable_oov')
oov2syllables_gen_path = 'dicts/syllable_oov'


def oov2syllables(oov2syllables_path):
    oov2syllables_dict = {}
    with open(oov2syllables_path) as f:
        data = [line.strip() for line in f]
        for line in data:
            if line and line[0] != "#" and "[" not in line:
                temp = (re.sub(" +", " ", line)).split(" ")
                if len(temp) > 1:
                    word, syllables = temp[0], temp[1:]
                    oov2syllables_dict[word] = syllables
    return oov2syllables_dict


def syllable2phonemes(phone_vn_path):
    syllable2phonemes_dict = {}
    with open(phone_vn_path) as f:
        data = [line.strip() for line in f]
        for line in data:
            if line and line[0] != "#":
                temp = line.strip().split(" ")
                syllable, phonemes = temp[0], temp[1:]
                syllable2phonemes_dict[syllable] = phonemes
    return syllable2phonemes_dict


def get_oov_phone_dict(oov2syllables_path=oov2syllables_gen_path, phone_vn_path=hp.phone_vn_train):
    oov2syllables_dict = oov2syllables(oov2syllables_path)
    syllable2phonemes_dict = syllable2phonemes(phone_vn_path)
    if oov2syllables_dict:
        phone_oov_dict_path = os.path.join(os.path.dirname(oov2syllables_path), 'phone_oov')
        phone_oov_dict_file = open(phone_oov_dict_path, 'w')
        oov2syllables_error = ''
        for oov_word in oov2syllables_dict:
            phone_list = []
            for syllable in oov2syllables_dict[oov_word]:
                if syllable not in syllable2phonemes_dict.keys():
                    oov2syllables_error += oov_word + " " + " ".join(oov2syllables_dict[oov_word]) + "\n"
                    phone_list = []
                    break
                else:
                    phone_list.append(" ".join(syllable2phonemes_dict[syllable]))
                    continue
            if phone_list:
                phone_oov_dict_file.write(oov_word + " " + " ".join(phone_list) + "\n")
        print(
            f'Created phone_oov at {os.path.dirname(oov2syllables_path)}')

        if oov2syllables_error:
            oov2syllables_error_path = oov2syllables_path.replace('.txt', '_error.txt')
            with open(oov2syllables_error_path, 'w') as f:
                f.write(oov2syllables_error)
            print(
                f'Created {os.path.basename(oov2syllables_error_path)} at {os.path.dirname(oov2syllables_path)}')


if __name__ == '__main__':
    get_oov_phone_dict()