#!/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2021.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File Name: transnet_spliter.py
Author: Wang Zhihua <wangzhihua@.com>
Create Time: 2021/06/19 15:16:25
Brief:      
"""

import sys
import os
import logging
import time
import json
import datetime
from utils import utils
import cv2
from tqdm import tqdm
import tensorflow as tf

class TransnetSpliter(object):
    def __init__(self, batch_size, device, use_gpu = True):
        self.batch_size = batch_size
        self.device = device
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        if use_gpu:
            config.gpu_options.allow_growth = True
        self.error_log = open('/home/tione/notebook/VideoStructuring/PipeLine/err.log', 'w')

    #根据frames分割video成mp4和wav
    def split_video(self, video_file, frames, split_dir, frame_count, fps):
        split_video_files = []
        split_audio_files = []
        for i, (start_index, end_index) in enumerate(utils.gen_ts_interval(frame_count, frames)):
            split_video_file, split_audio_file = utils.split_video(i, start_index, end_index, fps, video_file, split_dir, log = self.error_log)
            split_video_files.append(split_video_file)
            split_audio_files.append(split_audio_file)
        return split_video_files, split_audio_files


    def split(self, file_path, split_dir, shot_dir, fps):
        cap = cv2.VideoCapture(video_file)
        frame_count, fps, h, w = utils.read_video_info(cap)
        cap.release()

        frames = utils.get_frames_same_interval(frame_count, sample_fps)
        split_video_files, split_audio_files = self.split_video(video_file, frames, split_dir, frame_count, fps)
        vid = file_path.split('/')[-1].split('.')[0]
        with open(shot_dir + '/{}.info'.format(vid), 'w') as f:
            f.write(' '.join([str(frame) for frame in frames]) + '\n')
            f.write('\t'.join(split_video_files) + '\n')
            f.write('\t'.join(split_audio_files) + '\n')

