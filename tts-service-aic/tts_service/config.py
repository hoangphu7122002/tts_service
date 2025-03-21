import os

FLASK_SERVER_CONFIG = {
    'HOST': '0.0.0.0',
    "PORT": 8778,
    'DEBUG': False,
    'TESTING': False,
    # 'ENV': 'production',
    'ENV': 'development',
    'JSON_SORT_KEYS': False,
    'JSON_AS_ASCII': False
}

NORM_URL = "http://10.240.187.16:9779/normalize"
DOMAIN = "http://10.240.187.16:" + str(FLASK_SERVER_CONFIG["PORT"])

STORAGE_DATA = 'data'
STORAGE_CHUNK = 'chunk'

WORKING_DIR = os.path.dirname(__file__)
SITE_ROOT = os.path.join(WORKING_DIR, "../")

# API Config
API_VERSION = "1.0"

API_TIMEOUT = 60
MIN_TIMEOUT = 15


class GENDER:
    UNKNOWN = "UNKNOWN"
    MALE = "MALE"
    FEMALE = "FEMALE"


class REGION:
    UNKNOWN = "UNKNOWN"
    NORTH = "NORTH"
    SOUTH = "SOUTH"
    MIDDLE = "MIDDLE"


VOICE_CONFIGS = [
    {
        "id": 0,
        "info": {
            "id": 0,
            "name": "Test",
            "gender": GENDER.UNKNOWN,
            "region": REGION.UNKNOWN,
        },
        "storage_dir": "end2end_test",
        "enabled": True,
        "request_queue_name": "end2end_test_queue",
        "k_timeout": 10,
        "k_abnormal": 0.33,
    },
    {
        "id": 3,
        "info": {
            "id": 3,
            "name": "Hoa Mai - Sài Gòn",
            "gender": GENDER.FEMALE,
            "region": REGION.SOUTH
        },
        "storage_dir": "end2end_camhieu",
        "enabled": True,
        "request_queue_name": "end2end_camhieu_queue",
        "k_timeout": 0.12,
        "k_abnormal": 0.33,
    },
    {
        "id": 4,
        "info": {
            "id": 4,
            "name": "Hồng Đào - Hà Nội",
            "gender": GENDER.FEMALE,
            "region": REGION.NORTH
        },
        "storage_dir": "end2end_luongthuhien",
        "enabled": True,
        "request_queue_name": "end2end_luongthuhien_queue",
        "k_timeout": 0.24,
        "k_abnormal": 0.33,
    },
    {
        "id": 5,
        "info": {
            "id": 5,
            "name": "Nam An - Sài Gòn",
            "gender": GENDER.MALE,
            "region": REGION.SOUTH
        },
        "storage_dir": "end2end_thonghoaithanh",
        "enabled": True,
        "request_queue_name": "end2end_thonghoaithanh_queue",
        "k_timeout": 0.24,
        "k_abnormal": 0.33,
    },
    {
        "id": 6,
        "info": {
            "id": 6,
            "name": "Bắc Sơn - Hà Nội",
            "gender": GENDER.MALE,
            "region": REGION.NORTH
        },
        "storage_dir": "end2end_phananhgiap",
        "enabled": True,
        "request_queue_name": "end2end_phananhgiap_queue",
        "k_timeout": 0.24,
        "k_abnormal": 0.33,
    },
    {
        "id": 7,
        "info": {
            "id": 7,
            "name": "Hương Sen - Huế",
            "gender": GENDER.FEMALE,
            "region": REGION.MIDDLE
        },
        "storage_dir": "end2end_halinh",
        "enabled": True,
        "request_queue_name": "end2end_halinh_queue",
        "k_timeout": 0.24,
        "k_abnormal": 0.33,
    },
    {
        "id": 8,
        "info": {
            "id": 8,
            "name": "Trung Hà - Hà Tĩnh",
            "gender": GENDER.MALE,
            "region": REGION.MIDDLE
        },
        "storage_dir": "end2end_nguyensinhthanh",
        "enabled": True,
        "request_queue_name": "end2end_nguyensinhthanh_queue",
        "k_timeout": 0.24,
        "k_abnormal": 0.33,
    },
    {
        "id": 9,
        "info": {
            "id": 9,
            "name": "Thu Ngân - Huế",
            "gender": GENDER.FEMALE,
            "region": REGION.MIDDLE
        },
        "storage_dir": "end2end_thungan",
        "enabled": True,
        "request_queue_name": "end2end_thungan_queue",
        "k_timeout": 0.24,
        "k_abnormal": 0.33,
    },
    {
        "id": 10,
        "info": {
            "id": 10,
            "name": "Đào Hồng",
            "gender": GENDER.FEMALE,
            "region": REGION.NORTH
        },
        "storage_dir": "end2end_daohong",
        "enabled": True,
        "request_queue_name": "end2end_daohong_queue",
        "k_timeout": 0.3,
        "k_abnormal": 0.33,
    },
    {
        "id": 11,
        "info": {
            "id": 11,
            "name": "Hoa Mai",
            "gender": GENDER.FEMALE,
            "region": REGION.NORTH
        },
        "storage_dir": "end2end_hoamai_rt",
        "enabled": True,
        "request_queue_name": "end2end_hoamai_rt_queue",
        "k_timeout": 0.4,
        "k_abnormal": 0.4,
    },
    {
        "id": 12,
        "info": {
            "id": 12,
            "name": "Hồng Đào",
            "gender": GENDER.FEMALE,
            "region": REGION.SOUTH
        },
        "storage_dir": "end2end_hongdao_rt",
        "enabled": True,
        "request_queue_name": "end2end_hongdao_rt_queue",
        "k_timeout": 0.4,
        "k_abnormal": 0.4,
    },
    {
        "id": 13,
        "info": {
            "id": 13,
            "name": "Hương Sen",
            "gender": GENDER.FEMALE,
            "region": REGION.MIDDLE
        },
        "storage_dir": "end2end_huongsen_rt",
        "enabled": True,
        "request_queue_name": "end2end_huongsen_rt_queue",
        "k_timeout": 0.4,
        "k_abnormal": 0.4,
    },
]
