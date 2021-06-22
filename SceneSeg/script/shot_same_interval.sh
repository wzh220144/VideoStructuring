#!/bin/bash
#########################################################################
# Author: Wang Zhihua <wangzhihua@.com>
# Created Time: 2021/06/22 00:26:42
# File Name: shot_same_interval.sh
# Brief: 
# Version: 1.0
#########################################################################
python -u /home/tione/notebook/VideoStructuring/SceneSeg/pre/ShotDetect/shotdetect_same_interval.py --video_dir=/home/tione/notebook/dataset/videos/train_5k_A --save_dir=/home/tione/notebook/dataset/train_5k_A/shot_same_interval
python -u /home/tione/notebook/VideoStructuring/SceneSeg/pre/ShotDetect/shotdetect_same_interval.py --video_dir=/home/tione/notebook/dataset/videos/test_5k_2nd --save_dir=/home/tione/notebook/dataset/test_5k_2nd/shot_same_interval
