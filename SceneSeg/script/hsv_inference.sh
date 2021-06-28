#!/bin/bash
#########################################################################
# Author: Wang Zhihua <wangzhihua@.com>
# Created Time: 2021/06/23 09:52:54
# File Name: hsv_inference.sh
# Brief: 
# Version: 1.0
#########################################################################
cd /home/tione/notebook/VideoStructuring/SceneSeg
python -u /home/tione/notebook/VideoStructuring/SceneSeg/lgss/run_inference.py --config=/home/tione/notebook/VideoStructuring/SceneSeg/lgss/config/inference_hsv.py
