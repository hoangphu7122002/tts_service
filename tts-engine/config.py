from trained.yenchibi.hparams import hparams as hparams_yenchibi
from trained.ngocmiu.hparams import hparams as hparams_ngocmiu
from trained.camhieu.hparams import hparams as hparams_camhieu
from trained.luongthuhien.hparams import hparams as hparams_luongthuhien
from trained.thonghoaithanh.hparams import hparams as hparams_thonghoaithanh

SIGMA_WAVEGLOW = 0.666
DENOISER_STRENGTH = 0.01
USE_G2P = False

READABLE_OOV_DICT_PATH = 'trained/dicts/readable_oov_dict'

WAVEGLOW_PATHS = {
    "end2end_yenchibi": "/data/thangpn/ws/tts/TRAINED_DATA/zwaveglow/waveglow_v3_340k",
    "end2end_ngocmiu": "/data/thangpn/ws/tts/TRAINED_DATA/zwaveglow/waveglow_v3_340k",
    "end2end_camhieu": "/trained-models/waveglow/waveglow_luongthuhien-22k_v1_425k.pt",
    "end2end_luongthuhien": "/trained-models/waveglow/waveglow_luongthuhien-22k_v1_425k.pt",
    "end2end_thonghoaithanh": "/data/thangpn/ws/tts/TRAINED_DATA/zwaveglow/waveglow_thonghoaithanh-22k_v1-20191021_495k",
    "end2end_phananhgiap": "/data/thangpn/ws/tts/TRAINED_DATA/zwaveglow/waveglow_thonghoaithanh-22k_v1-20191021_495k",
    "end2end_halinh": "/trained-models/waveglow/waveglow_luongthuhien-22k_v1_425k.pt",
    "end2end_nguyensinhthanh": "/data/thangpn/ws/tts/TRAINED_DATA/zwaveglow/waveglow_thonghoaithanh-22k_v1-20191021_495k",
    "end2end_thungan": "/data/thangpn/ws/tts/TRAINED_DATA/zwaveglow/waveglow_luongthuhien-22k_v1_425k.pt",
    "end2end_daohong": "/data/thangpn/ws/tts/TRAINED_DATA/zwaveglow/waveglow_luongthuhien-22k_v1_425k.pt",
    "end2end_hoamai_rt": "/trained-models/waveglow/waveglow_luongthuhien-22k_v1_425k.pt",
    "end2end_hongdao_rt": "/trained-models/waveglow/waveglow_luongthuhien-22k_v1_425k.pt",
    "end2end_huongsen_rt": "/trained-models/waveglow/waveglow_luongthuhien-22k_v1_425k.pt",
    "end2end_test": "/data/thangpn/ws/tts/TRAINED_DATA/zwaveglow/waveglow_luongthuhien-22k_v1_425k.pt",
}
HPARAMS = {
    "end2end_yenchibi": hparams_yenchibi,
    "end2end_ngocmiu": hparams_ngocmiu,
    "end2end_camhieu": hparams_camhieu,
    "end2end_luongthuhien": hparams_luongthuhien,
    "end2end_thonghoaithanh": hparams_thonghoaithanh,
}

TACOTRON_PATHS = {
    "end2end_yenchibi": 'trained/yenchibi/tacotron_model',
    "end2end_ngocmiu": "trained/ngocmiu/tacotron_model",
    "end2end_camhieu": "/trained-models/tacotron/camhieu/tacotron2_camhieu_22k_v1-200110_120k.pt",
    "end2end_luongthuhien": "/trained-models/tacotron/luongthuhien/tacotron2_luongthuhien_22k_v2-191217_165k.pt",
    "end2end_thonghoaithanh": "/data/thangpn/ws/tts/TRAINED_DATA/thonghoaithanh/tacotron2_thonghoaithanh-22k_v2-tempo1.1-20191111_95k.pt",
    "end2end_phananhgiap": "/data/thangpn/ws/tts/TRAINED_DATA/phananhgiap/tacotron2_phananhgiap-22k_v4_74k.pt",
    "end2end_halinh": "/trained-models/tacotron/halinh/tacotron2_halinh-22k_v3-20200114_120k.pt",
    "end2end_nguyensinhthanh": "/data/thangpn/ws/tts/TRAINED_DATA/nguyensinhthanh/tacotron2_nguyensinhthanh-22k_v3-20191128_40k.pt",
    "end2end_thungan": "/data/storage/storage-tacotron/experiments/halinh-22k/checkpoints/v1-20191126/tacotron2_halinh-22k_v1-20191126_45k.pt",
    "end2end_daohong": "/data/storage/storage-tacotron/experiments/luongthuhien_22k/checkpoints/v2-191217/tacotron2_luongthuhien_22k_v2-191217_165k.pt",
    "end2end_hoamai_rt": "/trained-models/tacotron/camhieu/tacotron2_camhieu_22k_v1-200110_120k.pt",
    "end2end_hongdao_rt": "/trained-models/tacotron/luongthuhien/tacotron2_luongthuhien_22k_v2-191217_165k.pt",
    "end2end_huongsen_rt": "/trained-models/tacotron/halinh/tacotron2_halinh-22k_v3-20200114_120k.pt",
    "end2end_test": "trained/camhieu/tacotron2_camhieu-22k_v4_51k.pt",
}

QUEUES = {
    "end2end_yenchibi": "end2end_yenchibi_queue",
    "end2end_ngocmiu": "end2end_ngocmiu_queue",
    "end2end_camhieu": "end2end_camhieu_queue",
    "end2end_luongthuhien": "end2end_luongthuhien_queue",
    "end2end_thonghoaithanh": "end2end_thonghoaithanh_queue",
    "end2end_phananhgiap": "end2end_phananhgiap_queue",
    "end2end_halinh": "end2end_halinh_queue",
    "end2end_nguyensinhthanh": "end2end_nguyensinhthanh_queue",
    "end2end_thungan": "end2end_thungan_queue",
    "end2end_daohong": "end2end_daohong_queue",
    "end2end_hoamai_rt": "end2end_hoamai_rt_queue",
    "end2end_hongdao_rt": "end2end_hongdao_rt_queue",
    "end2end_huongsen_rt": "end2end_huongsen_rt_queue",
    "end2end_test": "end2end_test_queue",
}

MAX_WORDS = {
    "end2end_yenchibi": 20,
    "end2end_ngocmiu": 30,
    "end2end_camhieu": 45,
    "end2end_luongthuhien": 45,
    "end2end_thonghoaithanh": 30,
    "end2end_phananhgiap": 30,
    "end2end_halinh": 45,
    "end2end_nguyensinhthanh": 30,
    "end2end_thungan": 30,
    "end2end_daohong": 45,
    "end2end_hoamai_rt": 45,
    "end2end_hongdao_rt": 45,
    "end2end_huongsen_rt": 45,
    "end2end_test": 30,
}

USE_EOS = {
    "end2end_yenchibi": False,
    "end2end_ngocmiu": False,
    "end2end_camhieu": True,
    "end2end_luongthuhien": False,
    "end2end_thonghoaithanh": False,
    "end2end_phananhgiap": False,
    "end2end_halinh": False,
    "end2end_nguyensinhthanh": False,
    "end2end_thungan": False,
    "end2end_daohong": False,
    "end2end_hoamai_rt": True,
    "end2end_hongdao_rt": False,
    "end2end_huongsen_rt": False,
    "end2end_test": False,
}
