#!/bin/bash
#########################################################################
# Author: Wang Zhihua <wangzhihua@tencent.com>
# Created Time: 2021/06/27 18:44:50
# File Name: same_interval_inference.sh
# Brief: 
# Version: 1.0
#########################################################################
cd /home/tione/notebook/VideoStructuring/SceneSeg
python -u /home/tione/notebook/VideoStructuring/SceneSeg/lgss/run_inference.py --config=/home/tione/notebook/VideoStructuring/SceneSeg/lgss/config/inference_same_interval.py
