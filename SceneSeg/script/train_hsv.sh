#!/bin/bash
#########################################################################
# Author: Wang Zhihua <wangzhihua@.com>
# Created Time: 2021/06/21 16:01:44
# File Name: script/train_hsv.sh
# Brief: 
# Version: 1.0
#########################################################################
cd /home/tione/notebook/VideoStructuring/SceneSeg
python -u /home/tione/notebook/VideoStructuring/SceneSeg/lgss/run.py --config=/home/tione/notebook/VideoStructuring/SceneSeg/lgss/config/train_hsv.py
