#!/bin/bash#
data="data/thonghoaithanh-22k"
wavs="wavs_22k"
find $data/$wavs/ -name "*.wav" -size +0k > $data/all.txt
find $data/$wavs/ -name "*.wav" -size +35k | tail -n+2 > $data/train.txt
find $data/$wavs/ -name "*.wav" -size +35k | head -n2 > $data/test.txt
