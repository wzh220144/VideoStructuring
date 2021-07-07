#!/bin/bash
#########################################################################
# Author: Wang Zhihua <wangzhihua@tencent.com>
# Created Time: 2021/06/28 17:36:54
# File Name: feat_extract_train.sh
# Brief: 
# Version: 1.0
#########################################################################
cd /home/tione/notebook/VideoStructuring/MultiModal-Tagging
python -u /home/tione/notebook/VideoStructuring/MultiModal-Tagging/scripts/feat_extract_main.py --files_dir=/home/tione/notebook/dataset/train_5k_A/split --feat_dir=/home/tione/notebook/dataset/train_5k_A/split_feats --max_worker=10 --extract_img --extract_video --image_batch_size=64 --use_gpu --cuda_devices=1 --log_file='/home/tione/notebook/VideoStructuring/MultiModal-Tagging/err.log2'

#python -u /home/tione/notebook/VideoStructuring/MultiModal-Tagging/scripts/feat_extract_main.py --files_dir=/home/tione/notebook/dataset/train_5k_A/split --feat_dir=/home/tione/notebook/dataset/train_5k_A/split_feats --max_worker=10 --extract_ocr --extract_asr --image_batch_size=64 --use_gpu --cuda_devices=0 --log_file='/home/tione/notebook/VideoStructuring/MultiModal-Tagging/err.log1'
