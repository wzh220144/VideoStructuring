#!/bin/bash
#########################################################################
# Author: Wang Zhihua <wangzhihua@.com>
# Created Time: 2021/06/25 13:34:55
# File Name: same_interval_feat_extract.sh
# Brief: 
# Version: 1.0
#########################################################################
python -u /home/tione/notebook/VideoStructuring/SceneSeg/pre/place/extract_feat.py --data_root=/home/tione/notebook/dataset/train_5k_A/shot_same_interval --use_gpu=1
python -u /home/tione/notebook/VideoStructuring/SceneSeg/pre/audio/extract_feat.py --data_root=/home/tione/notebook/dataset/train_5k_A/shot_same_interval
#python -u /home/tione/notebook/VideoStructuring/SceneSeg/pre/place/extract_feat.py --data_root=/home/tione/notebook/dataset/test_5k_2nd/shot_same_interval --use_gpu=1
#python -u /home/tione/notebook/VideoStructuring/SceneSeg/pre/audio/extract_feat.py --data_root=/home/tione/notebook/dataset/test_5k_2nd/shot_same_interval
