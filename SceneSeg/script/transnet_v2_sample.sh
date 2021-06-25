#!/bin/bash
#########################################################################
# Author: Wang Zhihua <wangzhihua@.com>
# Created Time: 2021/06/24 14:06:34
# File Name: transnet_v2_sample.sh
# Brief: 
# Version: 1.0
#########################################################################
python -u /home/tione/notebook/VideoStructuring/SceneSeg/pre/gt_generator.py --video_dir /home/tione/notebook/dataset/videos/train_5k_A --data_root /home/tione/notebook/dataset/train_5k_A/shot_transnet_v2 --input_annotation /home/tione/notebook/dataset/GroundTruth/train5k.txt
