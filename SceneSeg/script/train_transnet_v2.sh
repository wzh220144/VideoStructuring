#!/bin/bash
#########################################################################
# Author: Wang Zhihua <wangzhihua@.com>
# Created Time: 2021/06/21 16:07:57
# File Name: script/tran_transnet_v2.sh
# Brief: 
# Version: 1.0
#########################################################################
cd /home/tione/notebook/VideoStructuring/SceneSeg
python -u /home/tione/notebook/VideoStructuring/SceneSeg/lgss/run.py --config=/home/tione/notebook/VideoStructuring/SceneSeg/lgss/config/train_transnet_v2.py
