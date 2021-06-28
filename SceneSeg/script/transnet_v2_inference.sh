#!/bin/bash
#########################################################################
# Author: Wang Zhihua <wangzhihua@.com>
# Created Time: 2021/06/25 21:46:58
# File Name: transnet_v2_inference.sh
# Brief: 
# Version: 1.0
#########################################################################
cd /home/tione/notebook/VideoStructuring/SceneSeg
python -u /home/tione/notebook/VideoStructuring/SceneSeg/lgss/run_inference.py --config=/home/tione/notebook/VideoStructuring/SceneSeg/lgss/config/inference_transnet_v2.py
