#!/bin/bash
#########################################################################
# Author: Wang Zhihua <wangzhihua@tencent.com>
# Created Time: 2021/07/03 15:40:27
# File Name: feat_extract_val.sh
# Brief: 
# Version: 1.0
#########################################################################
cd /home/tione/notebook/VideoStructuring/MultiModal-Tagging
python -u /home/tione/notebook/VideoStructuring/MultiModal-Tagging/scripts/feat_extract_main.py --files_dir=/home/tione/notebook/dataset/train_5k_A/shot_transnet_v2/output --feat_dir=/home/tione/notebook/dataset/train_5k_A/shot_transnet_v2/output_feats --max_worker=10 --extract_ocr --extract_asr --image_batch_size=64 --use_gpu --cuda_devices=0 --log_file='/home/tione/notebook/VideoStructuring/MultiModal-Tagging/err.log1'

#python -u /home/tione/notebook/VideoStructuring/MultiModal-Tagging/scripts/feat_extract_main.py --files_dir=/home/tione/notebook/dataset/train_5k_A/shot_transnet_v2/output --feat_dir=/home/tione/notebook/dataset/train_5k_A/shot_transnet_v2/output_feats --max_worker=10 --extract_asr --image_batch_size=64 --use_gpu

#python -u /home/tione/notebook/VideoStructuring/MultiModal-Tagging/scripts/feat_extract_main.py --files_dir=/home/tione/notebook/dataset/train_5k_A/shot_transnet_v2/output --feat_dir=/home/tione/notebook/dataset/train_5k_A/shot_transnet_v2/output_feats --max_worker=10 --extract_ocr --image_batch_size=64 --use_gpu
#python -u /home/tione/notebook/VideoStructuring/MultiModal-Tagging/scripts/feat_extract_main.py --files_dir=/home/tione/notebook/dataset/train_5k_A/shot_transnet_v2/output --feat_dir=/home/tione/notebook/dataset/train_5k_A/shot_transnet_v2/output_feats --max_worker=10 --extract_video --image_batch_size=64 --use_gpu
