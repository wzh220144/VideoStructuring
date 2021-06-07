import os
import sys

def get_frames_same_interval(frame_count, interval):
    frames = set([])
    cur_frame = 0
    while cur_frame < frame_count:
        frames.add(cur_frame)
        cur_frame += interval
    if frame_count > 0:
        frames.add(frame_count - 1)
    return frames


def get_frames_n_split(frame_count, n):
    step = min(frame_count // n, 1)
    frames = set([])
    cur_frame = 0
    while cur_frame < frame_count:
        frames.add(cur_frame)
        cur_frame += step
    if frame_count > 0:
        frames.add(frame_count - 1)
    return frames


def get_all_frames(frame_count):
    return range(frame_count)

def get_frames_same_ts_interval(ts, interval, max_frame):
    frames = set([])
    cur_frame = 0
    index = 0
    frame_count = len(ts)
    while cur_frame < frame_count:
        t = ts[cur_frame]
        if index * interval <= t:
            frames.add(cur_frame)
            index += 1
        cur_frame += 1
    return frames
