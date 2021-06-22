#!/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2021.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File Name: a.py
Author: Wang Zhihua <wangzhihua@.com>
Create Time: 2021/06/22 13:39:10
Brief:      
"""

import os
import cv2
import glob
import json
import tqdm

def gen_info(path):
    with open(path + '.info', 'w') as f:
        for x in tqdm.tqdm(glob.glob('{}/*.mp4'.format(path))):
            cap = cv2.VideoCapture(x)
            res = {}
            video_id = x.split('/')[0].split('.')[0]
            res[video_id] = {'frame_count': cap.get(cv2.CAP_PROP_FRAME_COUNT), 'fps': cap.get(cv2.CAP_PROP_FPS)}
        json.dump(res, f, ensure_ascii=False, indent = 4)


path = '/home/tione/notebook/VideoStructuring/dataset/videos/train_5k_A'
gen_info(path)
path = '/home/tione/notebook/VideoStructuring/dataset/videos/test_2k_2nd'
gen_info(path)
