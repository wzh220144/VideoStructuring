#!/bin/bash
#########################################################################
# Author: Wang Zhihua <wangzhihua@.com>
# Created Time: 2021/06/21 11:38:59
# File Name: shot_run.sh
# Brief: 
# Version: 1.0
#########################################################################
#python -u /home/tione/notebook/VideoStructuring/SceneSeg/pre/ShotDetect/shotdetect_hsv.py --video_dir=/home/tione/notebook/dataset/videos/train_5k_A --save_dir=/home/tione/notebook/dataset/train_5k_A/shot_hsv
python -u /home/tione/notebook/VideoStructuring/SceneSeg/pre/ShotDetect/shotdetect_hsv.py --video_dir=/home/tione/notebook/dataset/videos/test_5k_2nd --save_dir=/home/tione/notebook/dataset/test_5k_2nd/shot_hsv
#python -u /home/tione/notebook/VideoStructuring/SceneSeg/pre/ShotDetect/shotdetect_transnet_v2.py --video_dir=/home/tione/notebook/dataset/videos/train_5k_A --save_dir=/home/tione/notebook/dataset/train_5k_A/shot_transnet_v2
python -u /home/tione/notebook/VideoStructuring/SceneSeg/pre/ShotDetect/shotdetect_transnet_v2.py --video_dir=/home/tione/notebook/dataset/videos/test_5k_2nd --save_dir=/home/tione/notebook/dataset/test_5k_2nd/shot_transnet_v2
