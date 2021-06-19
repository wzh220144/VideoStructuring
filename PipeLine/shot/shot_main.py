#!/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2021.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File Name: shot_main.py
Author: Wang Zhihua <wangzhihua@.com>
Create Time: 2021/06/19 14:31:56
Brief:      
"""

import sys, os
sys.path.append(os.getcwd())

import argparse
import tqdm
import glob
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from shot.same_interval_spliter import SameIntervalSpliter
from shot.transnet_spliter import TransnetSpliter
from shot.hsv_spliter import HSVSpliter
import time
import tensorflow as tf
import random

def process_file(index, device, gens, file_path, shot_dir, fps):
    if not os.path.exists(file_path):
        return
    try:
        gen = gens[t]
        with tf.device('/gpu:{}'.format(device)):
            gen.split(file_path, shot_dir, fps)
    except Exception as e:
        print(file_path, traceback.format_exc())

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--files_dir', default='/home/tione/notebook/dataset/videos/train_5k_A', type=str)
    parser.add_argument('--postfix', default='mp4', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--feat_dir', default='/home/tione/notebook/dataset/feats/train_5k_A')
    parser.add_argument('--shot_dir', default='/home/tione/notebook/dataset/shot/train_5k_A')
    parser.add_argument('--max_worker', type=int, default=30)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--device', type=str, default='0,1')
    parser.add_argument('--mode', type=int, default=1)    #1: same interval, 2: shot with transnet, 3: shot with hsv
    args = parser.parse_args()

    shot_dir = args.shot_dir

    if args.mode == 1:
        shot_dir = args.shot_dir + '/same_interval'
    elif args.mode == 2:
        shot_dir = args.shot_dir + '/transnet'
    elif args.mode == 3:
        shot_dir = args.shot_dir + '/hsv'

    print(args.split_dir)

    os.makedirs(split_dir, exist_ok=True)
    os.makedirs(shot_dir, exist_ok=True)

    gens = []
    os.environ["CUDA_VISIBLE_DEVICES"]=args.device
    devices = args.device.split(',')
    for device in devices:
        with tf.device('/gpu:{}'.format(device)):
            if args.mode == 1:
                gens.append(SameIntervalSpliter(batch_size=args.batch_size, device='cuda:{}'.format(device)))
            elif args.mode == 2:
                gens.append(TransnetSpliter(batch_size=args.batch_size, device='cuda:{}'.format(device)))
            elif args.mode == 3:
                gens.append(HsvSpliter(batch_size=args.batch_size, device='cuda:{}'.format(device)))

    file_paths = glob.glob(args.files_dir+'/*.'+args.postfix)
    random.shuffle(file_paths)
    print('start split videos')
    with ThreadPoolExecutor(max_workers=args.max_worker) as executor:
        ps = []
        for file_path in file_paths:
            t = random.randint(0, len(gens) - 1)
            ps.append(executor.submit(process_file, t, devices[t], gens, file_path, shot_dir, args.fps))
        for p in tqdm.tqdm(ps, total=len(ps), desc='feat extract'):
            p.result()
            end_time = time.time()
