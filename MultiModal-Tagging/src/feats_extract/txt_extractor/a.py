#!/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2021tencent.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File Name: a.py
Author: Wang Zhihua <wangzhihua@tencent.com>
Create Time: 2021/05/23 00:03:28
Brief:      
"""

import sys
import os
import logging
import time
import json
import datetime
import cv2


def frame_iterator_list(filename, every_ms=1000, max_num_frames=300):
    video_capture = cv2.VideoCapture()
    if not video_capture.open(filename):
        print(sys.stderr, 'Error: Cannot open video file ' + filename)
        return
    last_ts = -99999  # The timestamp of last retrieved frame.
    num_retrieved = 0
    frame_all = []
    while num_retrieved < max_num_frames:
        while video_capture.get(0) < every_ms + last_ts:
            if not video_capture.read()[0]:
                return frame_all
        last_ts = video_capture.get(0)
        has_frames, frame = video_capture.read()
        if not has_frames:
            break
        frame_all.append(frame[:, :, ::-1])
        num_retrieved += 1
    return frame_all

video_path = "/home/tione/notebook/VideoStructuring/dataset/structuring/test5k_split_video/59d2a63744fcc0d02ac2c097d6973eaf#04#32.958#35.167#24.mp4"
rgb_list = frame_iterator_list(video_path, every_ms=1000)
print(rgb_list[len(rgb_list)//2], type(rgb_list[len(rgb_list)//2]), rgb_list[len(rgb_list)//2].shape)


img_path = "/home/tione/notebook/VideoStructuring/dataset/structuring/test5k_split_video_feats/image_jpg/59d2a63744fcc0d02ac2c097d6973eaf#04#32.958#35.167#24.jpg"
t = cv2.imread(img_path)[:, :, ::-1]
print(t, type(t), t.shape)
