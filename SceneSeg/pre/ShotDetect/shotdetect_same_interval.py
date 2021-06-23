#!/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2021.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File Name: shotdetect_same_interval.py
Author: Wang Zhihua <wangzhihua@.com>
Create Time: 2021/06/21 23:55:47
Brief:      
"""

from __future__ import print_function
from utilis import mkdir_ifmiss
from utilis.package import *
import random

from shotdetect.video_manager import VideoManager
from shotdetect.shot_manager import ShotManager

# For content-aware shot detection:
from shotdetect.video_splitter import is_ffmpeg_available,split_video_ffmpeg
from shotdetect.keyf_img_saver import generate_images,generate_images_txt

import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import glob
import numpy as np
import os
import ffmpeg
import cv2

def get_shots_from_cuts(cut_list, base_timecode, num_frames, start_frame = 0):
    # shot list, where shots are tuples of (Start FrameTimecode, End FrameTimecode).
    shot_list = []
    if not cut_list:
        shot_list.append((base_timecode + start_frame, base_timecode + num_frames))
        return shot_list
    # Initialize last_cut to the first frame we processed,as it will be
    # the start timecode for the first shot in the list.
    last_cut = base_timecode + start_frame
    t = 0
    for cut in cut_list:
        t = cut
        cut = base_timecode + cut
        shot_list.append((last_cut, cut))
        last_cut = cut
    # Last shot is from last cut to end of video.
    if t >= num_frames - 1:
        shot_list.append((last_cut, base_timecode + num_frames))
    return shot_list

def get_cut_list(frame_count, interval):
    res = []
    start = interval
    while start < frame_count - 1:
        res.append(start)
        start += interval
    return res


def main(args, video_path, data_root):
    cap = cv2.VideoCapture(video_path)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    video_path = osp.abspath(video_path)
    video_prefix = video_path.split("/")[-1].split(".")[0]

    video_manager = VideoManager([video_path])
    base_timecode = video_manager.get_base_timecode()


    try:
        cut_list = get_cut_list(frame_count, args.interval)
        # Obtain list of detected shots.
        shot_list = get_shots_from_cuts(cut_list, base_timecode, int(frame_count))

        # Set downscale factor to improve processing speed.
        if args.keep_resolution:
            video_manager.set_downscale_factor(1)
        else:
            video_manager.set_downscale_factor()
        # Start video_manager.
        video_manager.start()

        # Each shot is a tuple of (start, end) FrameTimecodes.
        if args.print_result:
            print('List of shots obtained:')
            for i, shot in enumerate(shot_list):

                print(
                    'Shot %4d: Start %s / Frame %d, End %s / Frame %d' % (
                        i,
                        shot[0].get_timecode(), shot[0].get_frames(),
                        shot[1].get_timecode(), shot[1].get_frames(),))
        # Save keyf img for each shot
        if args.save_keyf:
            output_dir = osp.join(data_root, "shot_keyf", video_prefix)
            print(output_dir)
            generate_images(video_manager, shot_list, output_dir, num_images=1)
        
        # Save keyf txt of frame ind
        if args.save_keyf_txt:
            output_dir = osp.join(data_root, "shot_txt", "{}.txt".format(video_prefix))
            mkdir_ifmiss(osp.join(data_root, "shot_txt"))
            generate_images_txt(shot_list, output_dir, num_images=3)

        # Split video into shot video
        if args.split_video:
            output_dir = osp.join(data_root, "shot_split_video", video_prefix)
            if not len(shot_list) == len(glob.glob(output_dir+'/*.mp4')):
                split_video_ffmpeg([video_path], shot_list, output_dir, suppress_output=True)
            if not len(shot_list) == len(glob.glob(output_dir+'/*.mp4')):
                os.system(" ".join(["rm", "-rf", output_dir]))

    finally:
        video_manager.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Single Video ShotDetect")
    parser.add_argument('--video_dir', type=str, default='/home/tione/notebook/dataset/videos/train_5k_A')
    parser.add_argument('--save_dir', type=str, default="/home/tione/notebook/dataset/train_5k_A/shot_same_interval", help="path to the saved data")
    parser.add_argument('--print_result', type = bool, default=True) #action="store_true")
    parser.add_argument('--save_keyf', type = bool, default=True) #      action="store_true")
    parser.add_argument('--save_keyf_txt', type = bool, default=True) #  action="store_true")
    parser.add_argument('--split_video', type = bool, default=True) #    action="store_true")
    parser.add_argument('--keep_resolution', type = bool, default=False)
    parser.add_argument('--interval', type = float, default=10)

    args = parser.parse_args()
    
    '''
    for video_path in glob.glob(args.video_dir+'/*.mp4'):
        print("...cutting shots for ", video_path)
        video_id = video_path.split('/')[-1].split(".mp4")[0]
        main(args, video_path, args.save_dir)
    '''

    results = []

    with ThreadPoolExecutor(max_workers=30) as executor:
        for video_path in glob.glob(args.video_dir+'/*.mp4'):
            print("...cutting shots for ", video_path)
            video_id = video_path.split('/')[-1].split(".mp4")[0]
            results.append(executor.submit(main, args, video_path, args.save_dir))
            #main(args, video_path, args.save_dir)
        results = [res.result() for res in results]
     
