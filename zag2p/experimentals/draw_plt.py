#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @AUTHOR : thangdc94
# @Date : 2019/12/30
import pickle
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    with open("result_er.pkl", "rb") as fp:
        (per_dict, wer_dict) = pickle.load(fp)
        per_dict = np.array(list(per_dict.items()))
        wer_dict = np.array(list(wer_dict.items()))
        plt.plot(per_dict[:, 0], per_dict[:, 1], label="per")
        plt.plot(wer_dict[:, 0], wer_dict[:, 1], label="wer")
        plt.legend()
        plt.grid()
        plt.show()
