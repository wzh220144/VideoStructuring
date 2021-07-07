#!/bin/bash
#########################################################################
# Author: Wang Zhihua <wangzhihua@tencent.com>
# Created Time: 2021/07/04 08:56:51
# File Name: scripts/eval.sh
# Brief: 
# Version: 1.0
#########################################################################
cd /home/tione/notebook/VideoStructuring/MultiModal-Tagging
rm -rf /home/tione/notebook/dataset/train_5k_A/shot_transnet_v2/tag_results
python scripts/inference_for_structuring.py --test_dir=/home/tione/notebook/dataset/train_5k_A/shot_transnet_v2/output --output_json=/home/tione/notebook/dataset/train_5k_A/shot_transnet_v2/outjson.txt --feat_dir=/home/tione/notebook/dataset/train_5k_A/shot_transnet_v2/output_feats --output_base=/home/tione/notebook/dataset/train_5k_A/shot_transnet_v2/tag_results --model_pb=/home/tione/notebook/dataset/model/tag/export/step_9000_1.1801
python scripts/eval_structuring.py
