#coding=utf-8

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
    cap = cv2.VideoCapture(video_file)
    video_id = video_file.split('/')[-1].split('.')[0]
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ori_fps = cap.get(cv2.CAP_PROP_FPS)

    frames, split_video_files, split_audio_files, flag = utils.read_shot_info(video_file, args.shot_dir)

    cur_frame = 0
    info = {'index': [], 'ts': []}
    if extract_youtube8m:
        info['youtube8m'] = []
    if extract_stft:
        info['stft'] = []
    
    youtube8m_feat_dict = {}
    stft_feat_dict = {}
    for x in glob.glob(youtube8m_feats_dir + '/{}/*.npy'.format(video_id)):
        cols = x.split('/')[-1].split('.')[0].split('#')
        key = cols[1] + '#' + cols[2]
        youtube8m_feat_dict[key] = x.split('/')[-1]
    for x in glob.glob(stft_feats_dir + '/{}/*.npy'.format(video_id)):
        cols = x.split('/')[-1].split('.')[0].split('#')
        key = cols[1] + '#' + cols[2]
        stft_feat_dict[key] = x.split('/')[-1]

    sorted_frames = sorted(list(frames))

    for i in range(len(frames)-1):
        start_frame = sorted_frames[i]
        end_frame = sorted_frames[i + 1]
        key = '{}#{}'.format(start_frame, end_frame)
        flag = True
        youtube8m_feat_path = ''
        stft_feat_path = ''
        if extract_youtube8m:
            if key in youtube8m_feat_dict:
                youtube8m_feat_path = youtube8m_feat_dict[key]
            else:
                flag = False
                print('{} does not exist for youtube8m.'.format(key))

        if extract_stft:
            if key in stft_feat_dict:
                stft_feat_path = stft_feat_dict[key]
            else:
                flag = False
                print('{} does not exist for stft.'.format(key))

        info['index'].append(start_frame)
        info['ts'].append(end_frame / ori_fps)
        info['youtube8m'].append(youtube8m_feat_path)
        info['stft'].append(stft_feat_path)
        cur_frame += 1
    info['org_fps'] = ori_fps
    info['w'] = w
    info['h'] = h
    info['fps'] = fps
    info['id'] = video_id
    info['duration'] = frame_count / fps
    info['frames'] = frame_count
    cap.release()
    return info

def trans2id(labels, label_id_dict):
    res = []
    for label in labels:
        res.append(int(label_id_dict[label]))
    return res

def gen_labels(info, label_id_dict, video_annotation):
    flag = True
    l = len(info['index'])
    info['label'] = []
    info['tag_label'] = []
    if len(video_annotation) == 0:
        for i in range(l):
            info['label'].append(0)
            info['tag_label'].append([])
    else:
        annotations = video_annotation['annotations']
        annotation_index = 0
        if len(annotations) < 1:
            flag = False
            if annotations[0]['segment'][0] != 0:
                flag = False
        t = -1
        for i in range(l):
            ts = info['ts'][i]
            if annotation_index == -1:
                info['label'].append(0)
                info['tag_label'].append([])
            else:
                annotation = annotations[annotation_index]
                segment = annotation['segment']
                s = segment[0]
                e = segment[1]
                if e < s:
                    flag = False
                if ts >= e:
                    info['label'].append(1)
                    annotation_index += 1
                else:
                    info['label'].append(0)
                if annotation_index == len(annotations):
                    annotation_index = -1
                    t = i
                if annotation_index != -1:
                    info['tag_label'].append(trans2id(annotations[annotation_index]['labels'], label_id_dict))
                else:
                    info['tag_label'].append([])
        if t != -1:
            for i in range(t, l):
                info['label'][i] = 0    #最后一个场景没有切分点
    return info, flag

def do_gen_samples(args, info, window_size):
    l = len(info['index'])
    res = []
    for cur in range(1, l - 1): #第一个frame及最后一个frame不能作为准确sample
        sample = {}
        b1 = list(range(max(0, cur - window_size), cur))
        b2 = list(range(cur, min(cur + window_size, l)))
        while len(b1) < window_size:
            b1.append(cur - 1)
        while len(b2) < window_size:
            b2.append(l - 1)
        sample['index'] = cur
        sample['b1'] = {'youtube8m': [info['youtube8m'][x] for x in b1], 'stft': [info['stft'][x] for x in b1]}
        sample['b2'] = {'youtube8m': [info['youtube8m'][x] for x in b2], 'stft': [info['stft'][x] for x in b2]}
        sample['label'] = info['label'][cur]
        sample['tag_label'] = info['tag_label'][cur]
        sample['ts'] = info['ts'][cur]
        sample['id'] = info['id']
        res.append(sample)
    return res

def _gen_samples(args, video_file, fps, window_size, label_id_dict, youtube8m_feats_dir, stft_feats_dir, extract_youtube8m, extract_stft, annotation_dict):
    video_info_file = video_file.replace('.mp4', '.info')
    if os.path.exists(video_info_file):
        with open(video_info_file, 'r') as fs:
            info = json.load(fs)
    else:
        info = read_video_info(video_file, fps, youtube8m_feats_dir, stft_feats_dir, extract_youtube8m, extract_stft)
        with open(video_info_file, 'w') as fs:
            json.dump(info, fs, ensure_ascii=False, indent=4)
    #print(info)
    video_name = video_file.split('/')[-1]
    video_annotation = {}
    if video_name in annotation_dict:
        video_annotation = annotation_dict[video_name]
    info, flag = gen_labels(info, label_id_dict, video_annotation)
    #print(info)
    if not flag:
        return []
    return do_gen_samples(args, info, window_size)

def gen_samples(args, annotation_dict, label_id_dict, fps, window_size, feats_dir, postfix, extract_youtube8m, extract_stft):
    youtube8m_feats_dir = os.path.join(feats_dir, postfix, 'youtube8m')
    stft_feats_dir = os.path.join(feats_dir, postfix, 'stft')
    video_files = glob.glob(os.path.join(args.video_dir, postfix, '*.mp4'))
    ps = []
    res = []
    with ThreadPoolExecutor(max_workers=args.max_worker) as executor:
        for video_file in tqdm.tqdm(video_files, total = len(video_files), desc = 'send task to pool'):
            #ps.append(executor.submit(_gen_samples, args, video_file, fps, window_size, label_id_dict, youtube8m_feats_dir, stft_feats_dir, extract_youtube8m, extract_stft, annotation_dict))
            res.extend(_gen_samples(args, video_file, fps, window_size, label_id_dict, youtube8m_feats_dir, stft_feats_dir, extract_youtube8m, extract_stft, annotation_dict))
        '''
        for p in tqdm.tqdm(ps, total = len(ps), desc = 'gen samples'):
            res.extend(p.result())
        '''
    return res

def parse_label_id(path):
    res = {}
    with open(path, 'r') as fs:
        for line in fs:
            cols = line.strip('\n').split('\t')
            res[cols[0]] = cols[1]
    return res

def write_samples(samples, fs):
    for sample in samples:
        fs.write(json.dumps(sample, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_postfix', type = str, default = 'train_5k_A')
    parser.add_argument('--test_postfix', type = str, default = 'test_5k_A')
    parser.add_argument('--video_dir', type = str, default = "/home/tione/notebook/VideoStructuring/dataset/videos")
    parser.add_argument('--shot_dir', type = str, default = "/home/tione/notebook/dataset/shot/train_5k_A/same_interval")
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

    os.makedirs(args.samples_dir, exist_ok=True)
    label_id_dict = parse_label_id(args.label_id)
    annotation_dict = {}
    
    if args.mode == 1:
        with open('{}/{}'.format(args.result_dir, args.train_postfix)) as f:
            annotation_dict = json.loads(f.read())
        samples = gen_samples(args, annotation_dict, label_id_dict,
                         args.fps, args.window_size, args.feats_dir,
                         args.train_postfix,
                         args.extract_youtube8m, args.extract_stft)
        
        ids = set([])
        for sample in samples:
            ids.add(sample['id'])
        ids = list(ids)
        random.shuffle(ids)
        print('video num: {}'.format(len(ids)))
        val_len = int(len(ids) * args.ratio)
        train_len = len(ids) - val_len
        val_ids = set(ids[:val_len])
        train_ids = set(ids[val_len:])
        print('train video num: {}, val video num: {}'.format(train_len, val_len))
        train_samples = []
        val_samples = []
        for sample in samples:
            if sample['id'] in train_ids:
                train_samples.append(sample)
            else:
                val_samples.append(sample)
        random.shuffle(train_samples)
        with open(args.samples_dir + '/{}'.format(args.train_postfix), 'w') as scene_train_fs:
            write_samples(train_samples, scene_train_fs)
        with open(args.samples_dir + '/val_{}'.format(args.train_postfix), 'w') as scene_val_fs:
            write_samples(val_samples, scene_val_fs)
        with open(args.samples_dir + '/val_annotation.txt', 'w') as val_annotation_fs:
            obj = [annotation_dict[i + '.mp4'] for i in val_ids]
            json.dump(obj, val_annotation_fs, ensure_ascii=False)
    elif args.mode == 2:
        samples = gen_samples(args, {}, label_id_dict, 
                         args.fps, args.window_size, args.feats_dir,
                         args.test_postfix,
                         args.extract_youtube8m, args.extract_stft)
        with open(args.samples_dir + '/{}'.format(args.test_postfix), 'w') as scene_test_fs:
            write_samples(samples, scene_test_fs)

