#!/bin/bash
#########################################################################
# Author: Wang Zhihua <wangzhihua@tencent.com>
# Created Time: 2021/07/04 06:44:59
# File Name: a.sh
# Brief: 
# Version: 1.0
#########################################################################
cd /home/tione/notebook/VideoStructuring/MultiModal-Tagging

#python -u /home/tione/notebook/VideoStructuring/MultiModal-Tagging/scripts/feat_extract_main.py --files_dir=/home/tione/notebook/dataset/train_5k_A/split --feat_dir=/home/tione/notebook/dataset/train_5k_A/split_feats --max_worker=20 --extract_video --image_batch_size=64 --use_gpu

python -u /home/tione/notebook/VideoStructuring/MultiModal-Tagging/scripts/feat_extract_main.py --files_dir=/home/tione/notebook/dataset/test_5k_2nd/shot_transnet_v2/output --feat_dir=/home/tione/notebook/dataset/test_5k_2nd/shot_transnet_v2/split_feats --max_worker=20 --extract_video --image_batch_size=64 --use_gpu

#python -u /home/tione/notebook/VideoStructuring/MultiModal-Tagging/scripts/feat_extract_main.py --files_dir=/home/tione/notebook/dataset/train_5k_A/shot_transnet_v2/output --feat_dir=/home/tione/notebook/dataset/train_5k_A/shot_transnet_v2/output_feats --max_worker=20 --extract_video --extract_audio --extract_img --image_batch_size=64 --use_gpu
