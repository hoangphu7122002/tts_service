import os
import numpy as np
from hparams import data_dir_path, files_lists_dir_path, metadata_file_path, metadata_wer_file_path
from utils import load_audio_name_and_text
if not os.path.isdir(files_lists_dir_path):
    os.makedirs(files_lists_dir_path)

metadata_raw_file = f'{data_dir_path}/metadata_raw.txt'
dict_metadata_ori = {}
if os.path.isfile(metadata_raw_file):
    with open(metadata_raw_file) as f:
        lines = [line.strip().split('|') for line in f if line]
    for line in lines:
        wav_name = line[0]
        if '.wav' not in wav_name:
            wav_name = wav_name + '.wav'
        text = line[1]
        text_without_punc = ' '.join(text.replace('.', '').replace(',', '').split()).strip()
        dict_metadata_ori[wav_name] = text_without_punc
else:
    print('No raw metadata.txt of {} found'.format(data_dir_path))


alignments_dir_path = f'{data_dir_path}/alignments'
if os.path.isdir(alignments_dir_path):

    alignments_files_list = list(os.listdir(alignments_dir_path))
    alignments_files_list.sort()

    metadata = open(metadata_file_path, 'w')
    remove_bad_silence = open(f'{files_lists_dir_path}/remove_bad_silence', 'w')
    remove_syl_dur = open(f'{files_lists_dir_path}/remove_syl_dur', 'w')
    remove_non_fluency = open(f'{files_lists_dir_path}/remove_non_fluency', 'w')
    remove_many_silence = open(f'{files_lists_dir_path}/remove_many_silence', 'w')
    remove_wer = open(f'{files_lists_dir_path}/remove_wer', 'w')
    wer_file_and_text = load_audio_name_and_text(metadata_wer_file_path)
    wer_file_list = []
    for pair in wer_file_and_text:
        wer_wav_name = pair[0]
        wer_text = ' '.join(pair[1].replace('~', '').split()).strip()
        wer_file_list.append(wer_wav_name)
        metadata.write(wer_wav_name + '|' + wer_text + '\n')
        text_without_punc = ' '.join(
            wer_text.replace('~', '').replace('.', '').replace(',', '').replace('*', '').replace('_',
                                                                                                 '').strip().split())
        if dict_metadata_ori and wer_wav_name in dict_metadata_ori and text_without_punc != dict_metadata_ori[
            wer_wav_name]:
            remove_wer.write(wer_wav_name + '|' + text_without_punc + '|' + dict_metadata_ori[wer_wav_name] + '\n')
    dict_non_fluency = {}

    for file in alignments_files_list:
        file_path = os.path.join(alignments_dir_path, file)
        wav_name = file.replace('.txt', '.wav')
        with open(file_path) as f:
            lines = [line.strip().split('\t') for line in f if line]
        text = ' '
        if lines and wav_name not in wer_file_list:
            dict_syl_dur = {}
            start = float(lines[0][0])
            end = float(lines[0][1])
            word = lines[0][2]
            text += word
            dict_syl_dur[word] = end - start
            count_silence = 0
            count_bad_silence = 0
            max_silence = 0
            for line in lines[1:]:
                start = float(line[0])
                silence = round(start - end, 2)
                if silence > max_silence:
                    max_silence = silence
                end = float(line[1])
                word = line[2]
                syl_dur = end - start
                dict_syl_dur[word] = syl_dur
                if silence < 0:
                    print(wav_name)
                if silence > 0:
                    count_silence += 1
                    if 0.03 < silence <= 0.09:
                        count_bad_silence += 1
                    elif 0.09 < silence < 0.18:
                        word = '~ ' + word
                    elif 0.18 <= silence < 0.3:
                        word = ', ' + word
                    elif 0.3 <= silence < 0.42:
                        word = '. ' + word
                    elif 0.42 <= silence:
                        word = '* ' + word
                text += ' ' + word
            count_bad_syl_dur = 0
            for syl_dur in dict_syl_dur.values():
                if syl_dur < 0.06:
                    count_bad_syl_dur += 1
            std_syl_dur = np.std(list(dict_syl_dur.values()))
            mu_syl_dur = np.mean(list(dict_syl_dur.values()))
            num_syllables = len(lines)
            duration = round(float(lines[-1][1]) - float(lines[0][0]), 2)
            non_fluency = max_silence * mu_syl_dur
            num_silence_syllable_ratio = count_silence / num_syllables

            if count_bad_silence >= 2:
                remove_bad_silence.write(wav_name + '\n')

            if num_silence_syllable_ratio > 0.5:
                remove_many_silence.write(wav_name + '\n')

            if std_syl_dur > 0.15 or count_bad_syl_dur > 0:
                remove_syl_dur.write(wav_name + '\n')

            if non_fluency > 0.1:
                remove_non_fluency.write(wav_name + '\n')

            text_without_punc = ' '.join(
                text.replace('~', '').replace('.', '').replace(',', '').replace('*', '').replace('_',
                                                                                                 '').strip().split())
            if dict_metadata_ori and wav_name in dict_metadata_ori and text_without_punc != dict_metadata_ori[wav_name]:
                remove_wer.write(wav_name + '|' + text_without_punc + '|' + dict_metadata_ori[wav_name] + '\n')
            text = text + ' '
            if os.path.basename(data_dir_path) == 'vlsp2019':
                text = text.replace(' gas ', ' ga ').replace(' hoá ', ' hóa ')
                text = text.replace(' taxi ', ' tắc xi ').replace(' virus ', ' vi dút ')
                text = text.replace(' karaoke ', ' ca ra ô kê ').replace(' studio ', ' x tiu đi ô ').replace(' test ',' tét x ').replace(' s ', ' ét x ') \
                    .replace(' asean ', ' a xê an ').replace(' stress ', ' x trét x ')
            text = text.strip()
            metadata.write(wav_name + '|' + text + '\n')

        if not lines:
            print(file)
    os.system(f'sort {metadata_file_path} -o {metadata_file_path}')
else:
    print('No alignments directory of {} found'.format(data_dir_path))
