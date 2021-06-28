#!/bin/bash
#########################################################################
# Author: Wang Zhihua <wangzhihua@tencent.com>
# Created Time: 2021/06/28 10:24:36
# File Name: transnet2_v2_split.sh
# Brief: 
# Version: 1.0
#########################################################################
cd /home/tione/notebook/VideoStructuring/SceneSeg
python -u /home/tione/notebook/VideoStructuring/SceneSeg/lgss/run_split.py --config=/home/tione/notebook/VideoStructuring/SceneSeg/lgss/config/inference_transnet_v2.py --threshold=0.94
