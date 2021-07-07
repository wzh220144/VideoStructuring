#!/bin/bash
#########################################################################
# Author: Wang Zhihua <wangzhihua@tencent.com>
# Created Time: 2021/06/28 18:15:37
# File Name: feat_extract_test.sh
# Brief: 
# Version: 1.0
#########################################################################
cd /home/tione/notebook/VideoStructuring/MultiModal-Tagging
#python -u /home/tione/notebook/VideoStructuring/MultiModal-Tagging/scripts/feat_extract_main.py --files_dir=/home/tione/notebook/dataset/test_5k_2nd/shot_transnet_v2/output --feat_dir=/home/tione/notebook/dataset/test_5k_2nd/shot_transnet_v2/split_feats --max_worker=10 --extract_video --extract_img --extract_audio --image_batch_size=64 --use_gpu --cuda_devices=1 --log_file='/home/tione/notebook/VideoStructuring/MultiModal-Tagging/err.log2'
python -u /home/tione/notebook/VideoStructuring/MultiModal-Tagging/scripts/feat_extract_main.py --files_dir=/home/tione/notebook/dataset/test_5k_2nd/shot_transnet_v2/output --feat_dir=/home/tione/notebook/dataset/test_5k_2nd/shot_transnet_v2/split_feats --max_worker=10 --extract_asr --extract_ocr --image_batch_size=64 --use_gpu --cuda_devices=1 --log_file='/home/tione/notebook/VideoStructuring/MultiModal-Tagging/err.log2'
