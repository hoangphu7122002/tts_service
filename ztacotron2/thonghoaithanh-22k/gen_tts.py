from tts_infer import load_tacotron_model, text2mel, text2speech, save_wav, save_mel
import torch
from glow import Denoiser
from text_embedding import word2phone
import os
from ZaG2P.api import load_model
import numpy as np
import argparse
import time
import warnings
warnings.filterwarnings("ignore")

sigma = 0.666
denoiser_strength = 0.00
use_g2p = False
phone_oov_gen_dict_path = "dicts/phone_oov"

waveglow_model_16k = 'waveglow_models/waveglow_doanngocle_v2'
waveglow_model_22k = 'waveglow_models/waveglow_numiennam_v1'

test_text = 'nhưng hà đức chinh gây ra quá nhiều thất vọng , kỹ thuật tầm thường và lối chơi ỷ quá nhiều vào sức của đức chinh không đúng tầm của một tuyển thủ quốc gia , lại là tuyển thủ chơi ở vị trí tiền đạo'
test_file = 'test_set/vlsp2018.txt'

parser = argparse.ArgumentParser()
parser.add_argument('--gen_wav',  action='store_true', default=True)
parser.add_argument('--no-gen_wav', dest='gen_wav', action='store_false')
parser.add_argument('--gen_mel', action='store_true')
parser.add_argument('--tacotron2', '-t', type=str, default='last', help='tacotron2 model path')
parser.add_argument('--waveglow', '-w', type=str, default='', help='waveglow model path')
parser.add_argument('--sigma', type=float, default=sigma)
parser.add_argument('--denoise', type=float, default=denoiser_strength)
parser.add_argument('--use_g2p', action='store_true')
parser.add_argument('--phoneoov', type=str, default=phone_oov_gen_dict_path)
parser.add_argument('--test', type=str, default=test_file, help='test file path or text')
parser.add_argument('--cuda', '-c', type=str, default='0')
args = parser.parse_args()

waveglow_model_path = args.waveglow
tacotron2_model_path = args.tacotron2

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)

print('Preparing models ...')
if tacotron2_model_path == 'last':
    from hparams import create_hparams_and_paths
    hp, paths = create_hparams_and_paths('')
    part1 = 'tacotron2_'
    part3 = 'k.pt'
    part2s = [int(chp.replace(part1, '').replace(part3, '')) for chp in os.listdir(paths.tacotron_models_dir_path) if
              chp.startswith(part1)]
    tacotron2_model_path = os.path.join(paths.tacotron_models_dir_path, part1 + str(max(part2s))) + part3
    checkpoint_dict = torch.load(tacotron2_model_path)

else:
    checkpoint_dict = torch.load(tacotron2_model_path)
    hp = checkpoint_dict['hparams']
    from paths import Paths
    paths = Paths(hp.data, hp.version)


tacotron_model = load_tacotron_model(hp)
tacotron_model.load_state_dict(checkpoint_dict['state_dict'])
_ = tacotron_model.cuda().eval().half()

if not waveglow_model_path:
    if hp.sampling_rate == 16000:
        waveglow_model_path = waveglow_model_16k
    if hp.sampling_rate == 22050:
        waveglow_model_path = waveglow_model_22k

waveglow_model = torch.load(waveglow_model_path)['model']
waveglow_model.cuda().eval().half()
for k in waveglow_model.convinv:
    k.float()

denoiser_model = Denoiser(waveglow_model).cuda()

g2p_model, viet_dict = load_model()

os.makedirs(paths.samples_dir_path, exist_ok=True)
os.chmod(paths.samples_dir_path, 0o775)

output_dir_name = os.path.basename(tacotron2_model_path).replace('_', '_' + hp.version + '_').replace('.pt', '') + '+' + os.path.basename(waveglow_model_path)
output_dir_path = os.path.join(paths.samples_dir_path, output_dir_name)

os.makedirs(output_dir_path, exist_ok=True)
os.chmod(output_dir_path, 0o775)

print('Preparing dictionaries...')
vn2phone_train_dict = checkpoint_dict['vn2phone_train_dict']
oov2phone_train_dict = checkpoint_dict['oov2phone_train_dict']
oov2phone_gen_dict = word2phone(phone_oov_gen_dict_path, hp.coda_nucleus_and_semivowel)
if os.path.isfile(phone_oov_gen_dict_path):
    word2phone_dict = {**vn2phone_train_dict, **oov2phone_gen_dict}
else:
    word2phone_dict = {**vn2phone_train_dict, **oov2phone_train_dict}
symbol2numeric_dict = checkpoint_dict['symbol2numeric_dict']

print('Ready...')
if os.path.isfile(args.test):
    test_set_path, extension = os.path.splitext(args.test)
    test_set_name = os.path.basename(test_set_path)
    with open(args.test) as f:
        for line in f:
            file_name, text = line.strip().split('|')
            file_name = test_set_name + '_' + file_name
            if args.gen_wav:
                wav_save_path = os.path.join(output_dir_path, file_name)
                if '.wav' not in file_name:
                    wav_save_path = wav_save_path + '.wav'
                wav = text2speech(text, tacotron_model, waveglow_model, denoiser_model, g2p_model, viet_dict,
                                    word2phone_dict, symbol2numeric_dict,
                                    hp.p_phone_mix, hp.punctuation, hp.eos, use_g2p=args.use_g2p)
                save_wav(wav, wav_save_path, hp.sampling_rate)
            if args.gen_mel:

                mel_path = os.path.join(output_dir_path, file_name)
                mel = text2mel(text, tacotron_model, hp.p_phone_mix, g2p_model, viet_dict, word2phone_dict,
                               symbol2numeric_dict, hp.punctuation, hp.eos, use_g2p=use_g2p)
                mel = mel.squeeze().cpu().detach().numpy()
                np.save(mel_path, mel)

else:
    file_name = time.strftime("%H%M%S")
    if args.gen_wav:
        wav_save_path = os.path.join(output_dir_path, file_name + '.wav')
        wav = text2speech(args.test, tacotron_model, waveglow_model, denoiser_model, g2p_model, viet_dict,
                            word2phone_dict, symbol2numeric_dict,
                            hp.p_phone_mix, hp.punctuation, hp.eos, use_g2p=args.use_g2p, sigma=args.sigma, denoiser_strength=args.denoise)
        save_wav(wav, wav_save_path, hp.sampling_rate)
        #os.system(f'play {wav_save_path}')
    if args.gen_mel:
        mel_save_path = os.path.join(output_dir_path, file_name)
        mel = text2mel(args.test, tacotron_model, g2p_model, viet_dict, word2phone_dict,
                       symbol2numeric_dict, hp.p_phone_mix, hp.punctuation, hp.eos, use_g2p=use_g2p)
        mel = mel.squeeze().cpu().detach().numpy()
        save_mel(mel_save_path, mel)



