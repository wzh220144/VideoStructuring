#!/bin/bash
#########################################################################
# Author: Wang Zhihua <wangzhihua@tencent.com>
# Created Time: 2021/07/04 14:33:07
# File Name: b.sh
# Brief: 
# Version: 1.0
#########################################################################
cd /home/tione/notebook/VideoStructuring/MultiModal-Tagging

#rm -rf /home/tione/notebook/dataset/train_5k_A/split_feats/video_npy
#rm -rf /home/tione/notebook/dataset/test_5k_2nd/shot_transnet_v2/split_feats/video_npy
#rm -rf /home/tione/notebook/dataset/train_5k_A/shot_transnet_v2/output_feats/video_npy

python -u /home/tione/notebook/VideoStructuring/MultiModal-Tagging/scripts/feat_extract_main_tmp.py --max_worker=10 --extract_video --image_batch_size=64 --use_gpu
