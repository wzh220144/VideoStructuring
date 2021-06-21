#!/bin/bash
#########################################################################
# Author: Wang Zhihua <wangzhihua@.com>
# Created Time: 2021/06/19 15:14:33
# File Name: shot_split.sh
# Brief: 
# Version: 1.0
#########################################################################
cd /home/tione/notebook/VideoStructuring/PipeLine
#python -u /home/tione/notebook/VideoStructuring/PipeLine/shot/shot_main.py --mode=1
python -u /home/tione/notebook/VideoStructuring/PipeLine/shot/shot_main.py --mode=1 --files_dir=/home/tione/notebook/dataset/videos/test_5k_2nd --feat_dir=/home/tione/notebook/dataset/feats/test_5k_2nd --shot_dir=/home/tione/notebook/dataset/shot/test_5k_2nd
