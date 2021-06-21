import os
import sys
import cv2
import subprocess

def gen_ts_interval(frame_count, frames):
    count = 0
    if frame_count <= 0:
        return
    pre_index = 0
    cur_index = 0
    while True:
        cur_index += 1
        if cur_index >= frame_count:
            break
        if cur_index in frames:
            yield pre_index, cur_index
            count += 1
            pre_index = cur_index

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
    '''
    if frame_count > 0:
        frames.add(frame_count - 1)
    '''
    return frames


def get_all_frames(frame_count):
    return range(frame_count)

def read_video_info(cap):
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    return frame_count, fps, h, w

def trans2audio(video_file, audio_file):
    if not os.path.exists(audio_file):
        command = 'ffmpeg -loglevel error -i '+ video_file + ' ' + audio_file
    return os.system(command)

def format_time(ts):
    hrs = int(ts / 3600)
    ts -= (hrs * 3600)
    mins = int(ts / 60)
    ts -= (mins * 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hrs, mins, ts)

def split_video(i, start_frame, end_frame, fps, video_file, split_dir, log = None,
                       arg_override = '-crf 21', hide_progress = False, suppress_output = True):
    video_id = video_file.split('/')[-1].split('.')[0]
    split_dir = '{}/{}'.format(split_dir, video_id)
    os.makedirs(split_dir, exist_ok=True)
    split_video_file = '{}/{}#{}#{}#{}.mp4'.format(split_dir, i, start_frame, end_frame, int(fps))
    split_audio_file = '{}/{}#{}#{}#{}.wav'.format(split_dir, i, start_frame, end_frame, int(fps))
    flag = False
    if os.path.exists(split_video_file):
        if log != None:
            log.write('{} exist.\n'.format(split_video_file))
    else:
        if log != None:
            log.write('{} does not exist.\n'.format(split_video_file))
        duration = (end_frame - start_frame - 1) / fps
        call_list = ['ffmpeg']
        if suppress_output:
            call_list += ['-v', 'quiet']
        elif i > 0:
            call_list += ['-v', 'error']
        call_list += [
            '-y',
            '-ss',
            format_time(start_frame / fps),
            '-i',
            video_file
        ]
        call_list += arg_override.split(' ')    # compress
        call_list += ['-map_chapters', '-1']  # remove meta stream
        call_list += [
            '-strict',
            '-2',
            '-t',
            format_time(duration),
            '-sn',
            split_video_file]
        ret_val = subprocess.call(call_list)
        if ret_val != 0:
            if log != None:
                log.write('split {} failed.\n'.format(split_video_file))
    if os.path.exists(split_audio_file):
        if log != None:
            log.write('{} exist.\n'.format(split_audio_file))
    else:
        if not os.path.exists(split_video_file):
            if log != None:
                log.write('trans {} failed: {} does not exist.\n'.format(split_audio_file, split_video_file))
        else:
            ret = trans2audio(split_video_file, split_audio_file) == 0
            if not ret:
                log.write('trans {} failed, ret code: {}\n'.format(split_audio_file, ret))
    flag = os.path.exists(split_video_file) and os.path.exists(split_audio_file)
    return split_video_file, split_audio_file, flag

def read_shot_info(video_file, shot_dir):
    frames = set([])
    split_video_files = []
    split_audio_files = []
    flag = False
    video_id = video_file.split('/')[-1].split('.')[0]
    info_file = '{}/{}.info'.format(shot_dir, video_id)
    if os.path.exists(info_file):
        with open(info_file, 'r') as f:
            flag = True
            frames = set([int(x) for x in f.readline().strip('\n').split(' ')])
            split_video_files = f.readline().strip('\n').split('\t')
            split_audio_files = f.readline().strip('\n').split('\t')
    return frames, split_video_files, split_audio_files, flag
