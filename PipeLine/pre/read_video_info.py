#!/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2021tencent.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File Name: /home/tione/notebook/VideoStructuring/PipeLine/pre/read_video_info.py
Author: Wang Zhihua <wangzhihua@tencent.com>
Create Time: 2021/06/17 15:33:55
Brief:      
"""

import os
import argparse
import json
import cv2
import random
from utils import utils
import glob
import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time

def read_video_info(video_file, fps, youtube8m_feats_dir, stft_feats_dir, extract_youtube8m, extract_stft):
    begin = time.time()
    cap = cv2.VideoCapture(video_file)
    video_id = video_file.split('/')[-1].split('.')[0]
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ori_fps = cap.get(cv2.CAP_PROP_FPS)
    frames = utils.get_frames_same_interval(frame_count, fps)
    cur_frame = 0
    info = {'index': [], 'ts': []}
    if extract_youtube8m:
        info['youtube8m'] = []
    if extract_stft:
        info['stft'] = []
    duration = 0.0
    while True:
        has_frame, frame = cap.read()
        if not has_frame:
            break
        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        duration = max(duration, ts)
        if cur_frame in frames:
            flag = True
            youtube8m_feat_path = ''
            stft_feat_path = ''
            if extract_youtube8m:
                youtube8m_feat_path = os.path.join(youtube8m_feats_dir, '{}#{}.npy'.format(video_id, cur_frame))
                if not os.path.exists(youtube8m_feat_path):
                    #print('{} do not exist.'.format(youtube8m_feat_path))
                    flag = False
                youtube8m_feat_path = '#{}.npy'.format(cur_frame)
            if extract_stft:
                stft_feat_path = os.path.join(stft_feats_dir, '{}#{}#{}.npy'.format(video_id, cur_frame, cur_frame + fps))
                if not os.path.exists(stft_feat_path):
                    #print('{} do not exist.'.format(stft_feat_path))
                    flag = False
                stft_feat_path = '#{}#{}.npy'.format(cur_frame, cur_frame + fps)
            if not flag:
                cur_frame += 1
                continue
            info['index'].append(cur_frame)
            info['ts'].append(ts)
            if extract_youtube8m:
                info['youtube8m'].append(youtube8m_feat_path)
            if extract_stft:
                info['stft'].append(stft_feat_path)
        cur_frame += 1
    info['org_fps'] = ori_fps
    info['w'] = w
    info['h'] = h
    info['fps'] = fps
    info['id'] = video_id
    info['duration'] = frame_count / ori_fps
    info['frames'] = frame_count
    cap.release()
    end = time.time()
    duration =  float(frame_count) / float(ori_fps)
    return info

def _read_video_info(args, video_file, fps, window_size, label_id_dict, youtube8m_feats_dir, stft_feats_dir, extract_youtube8m, extract_stft, annotation_dict):
    video_info_file = video_file.replace('.mp4', '.info')
    if os.path.exists(video_info_file):
        with open(video_info_file, 'r') as fs:
            info = json.load(fs)
    else:
        info = read_video_info(video_file, fps, youtube8m_feats_dir, stft_feats_dir, extract_youtube8m, extract_stft)
        with open(video_info_file, 'w') as fs:
            json.dump(info, fs, ensure_ascii=False)

def gen_video_infos(args, annotation_dict, label_id_dict, fps, window_size, feats_dir, postfix, extract_youtube8m, extract_stft):
    youtube8m_feats_dir = os.path.join(feats_dir, postfix, 'youtube8m')
    stft_feats_dir = os.path.join(feats_dir, postfix, 'stft')
    video_files = glob.glob(os.path.join(args.video_dir, postfix, '*.mp4'))
    ps = []
    res = []
    with ThreadPoolExecutor(max_workers=args.max_worker) as executor:
        for video_file in tqdm.tqdm(video_files, total = len(video_files), desc = 'send task to pool'):
            ps.append(executor.submit(_read_video_info, args, video_file, fps, window_size, label_id_dict, youtube8m_feats_dir, stft_feats_dir, extract_youtube8m, extract_stft, annotation_dict))
        for p in tqdm.tqdm(ps, total = len(ps), desc = 'gen samples'):
            p.result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_postfix', type = str, default = 'train_5k_A')
    parser.add_argument('--test_postfix', type = str, default = 'test_5k_2nd')
    parser.add_argument('--video_dir', type = str, default = "/home/tione/notebook/VideoStructuring/dataset/videos")
    parser.add_argument('--data_root', type = str, default = "/home/tione/notebook/VideoStructuring/dataset")
    parser.add_argument('--train_txt', type = str, default = "/home/tione/notebook/VideoStructuring/dataset/structuring/GroundTruth/train5k.txt")
    parser.add_argument('--label_id', type = str, default = '/home/tione/notebook/VideoStructuring/dataset/label_id.txt')
    parser.add_argument('--feats_dir', type = str, default = '/home/tione/notebook/VideoStructuring/dataset/feats')
    parser.add_argument('--extract_youtube8m', type = bool, default = True)
    parser.add_argument('--extract_stft', type = bool, default = True)
    parser.add_argument('--fps', type = int, default = 5)
    parser.add_argument('--ratio', type = float, default = 0.04)
    parser.add_argument('--samples_dir', type = str, default = '/home/tione/notebook/VideoStructuring/dataset/samples/seg')
    parser.add_argument('--result_dir', type = str, default = '/home/tione/notebook/VideoStructuring/dataset/result/seg')
    parser.add_argument('--window_size', type = int, default = 5)
    parser.add_argument('--mode', type = int, default = 1)
    parser.add_argument('--max_worker', type = int, default = 10)
    args = parser.parse_args()

    annotation_dict = {}
 
    if args.mode == 1:
        gen_video_infos(args, annotation_dict, {},
                args.fps, args.window_size, args.feats_dir,
                args.train_postfix,
                args.extract_youtube8m, args.extract_stft)
    elif args.mode == 2:
        gen_video_infos(args, annotation_dict, {},
                args.fps, args.window_size, args.feats_dir,
                args.test_postfix,
                args.extract_youtube8m, args.extract_stft)

