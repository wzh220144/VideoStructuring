#!/bin/bash
#########################################################################
# Author: Wang Zhihua <wangzhihua@.com>
# Created Time: 2021/06/25 13:36:35
# File Name: same_train.sh
# Brief: 
# Version: 1.0
#########################################################################
cd /home/tione/notebook/VideoStructuring/SceneSeg
python -u /home/tione/notebook/VideoStructuring/SceneSeg/lgss/run.py --config=/home/tione/notebook/VideoStructuring/SceneSeg/lgss/config/train_same_interval.py
