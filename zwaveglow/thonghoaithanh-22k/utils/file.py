import torch
import numpy as np
import os
import csv
import sys
import math
from scipy.io.wavfile import read
import librosa


def files_to_list(filename):
    if os.path.splitext(filename)[1] != '.wav':
        with open(filename, encoding='utf-8') as f:
            files = f.readlines()
        files = [f.rstrip() for f in files if f]
    else:
        files = [filename]      
    return files
