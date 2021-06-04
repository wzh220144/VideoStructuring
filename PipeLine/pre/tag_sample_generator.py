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

def read_video_info(video_file, fps, youtube8m_feats_dir, stft_feats_dir, extract_youtube8m, extract_stft):
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
    while True:
        has_frame, frame = cap.read()
        if not has_frame:
            break
        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if cur_frame in frames:
            flag = True
            youtube8m_feat_path = ''
            stft_feat_path = ''
            if extract_youtube8m:
                youtube8m_feat_path = os.path.join(youtube8m_feats_dir, '{}#{}.npy'.format(video_id, cur_frame))
                if not os.path.exists(youtube8m_feat_path):
                    flag = False
                youtube8m_feat_path = '#{}.npy'.format(cur_frame)
            if extract_stft:
                stft_feat_path = os.path.join(stft_feats_dir, '{}{}#{}#{}.npy'.format(video_id, video_id, cur_frame, cur_frame + fps))
                if not os.path.exists(stft_feat_path):
                    flag = False
                stft_feat_path = '#{}#{}.npy'.format(cur_frame, cur_frame + fps)
            if not flag:
                break
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
            info['seg_index'].append(0)
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
                info['seg_index'].append(annotation_index)
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
                info['seg_index'].append(annotation_index)
        if t != -1:
            for i in range(t, l):
                info['label'][i] = 0    #最后一个场景没有切分点
    return info, flag

def do_gen_samples(info):
    l = len(info['index'])
    res = []
    pre_seg_index = -1
    cur_sample = {'index': [], 'youtube8m': [], 'stft': [], 'label': [], 'tag_label': [], 'ts': [], 'id': '', 'seg_index': -1}
    for cur in range(0, l):
        seg_index = info['seg_index'][cur]
        if seg_index == -1:
            continue
        if seg_index != pre_seg_index:
            if cur_sample['seg_index'] >= 0:
                res.append(cur_sample)
            pre_seg_index = seg_index
            cur_sample = {'index': [], 'youtube8m': [], 'stft': [], 'label': [], 'tag_label': [], 'ts': [], 'id': '', 'seg_index': seg_index}
        cur_sample['index'].append(cur)
        cur_sample['youtube8m'].append(info['youtube8m'][cur])
        cur_sample['stft'].append(info['stft'][cur])
        cur_sample['label'].append(info['label'][cur])
        cur_sample['tag_label'] = info['tag_label'][cur]
        cur_sample['ts'].append(info['ts'][cur])
        cur_sample['id'] = info['id']
    if len(cur_sample['seg_index']) > 0:
        res.append(cur_sample)
    return res

def _gen_samples(video_file, fps, label_id_dict, youtube8m_feats_dir, stft_feats_dir, extract_youtube8m, extract_stft, annotation_dict):
    info = read_video_info(video_file, fps, youtube8m_feats_dir, stft_feats_dir, extract_youtube8m, extract_stft)
    #print(info)
    video_name = video_file.split('/')[-1]
    video_annotation = {}
    if video_name in annotation_dict:
        video_annotation = annotation_dict[video_name]
    info, flag = gen_labels(info, label_id_dict, video_annotation)
    #print(info)
    if not flag:
        return []
    return do_gen_samples(info)

def gen_samples(annotation_dict, label_id_dict, fps, feats_dir, postfix, extract_youtube8m, extract_stft):
    youtube8m_feats_dir = os.path.join(feats_dir, postfix, 'youtube8m')
    stft_feats_dir = os.path.join(feats_dir, postfix, 'stft')
    res = []
    video_files = glob.glob(os.path.join(args.video_dir, postfix, '*.mp4'))
    ps = []
    with ThreadPoolExecutor(max_workers=args.max_worker) as executor:
        for video_file in tqdm.tqdm(video_files, total=len(video_files), desc='send task to pool'):
            ps.append(executor.submit(_gen_samples, video_file, fps, label_id_dict, youtube8m_feats_dir,
                                      stft_feats_dir, extract_youtube8m, extract_stft, annotation_dict))
        for p in tqdm.tqdm(ps, total=len(ps), desc='gen samples'):
            res.extend(p.result())
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
    parser.add_argument('--data_root', type = str, default = "/home/tione/notebook/VideoStructuring/dataset")
    parser.add_argument('--annotation_file', type = str, default = "/home/tione/notebook/VideoStructuring/dataset/structuring/GroundTruth/train5k.txt")
    parser.add_argument('--label_id', type = str, default = '/home/tione/notebook/VideoStructuring/dataset/label_id.txt')
    parser.add_argument('--feats_dir', type = str, default = '/home/tione/notebook/VideoStructuring/dataset/feats')
    parser.add_argument('--extract_youtube8m', type = bool, default = True)
    parser.add_argument('--extract_stft', type = bool, default = True)
    parser.add_argument('--fps', type = int, default = 5)
    parser.add_argument('--ratio', type = float, default = 0.05)
    parser.add_argument('--samples_dir', type = str, default = '/home/tione/notebook/VideoStructuring/dataset/samples/tag')
    parser.add_argument('--max_worker', type = int, default = 20)
    args = parser.parse_args()

    os.makedirs(args.samples_dir, exist_ok=True)
    label_id_dict = parse_label_id(args.label_id)
    annotation_dict = {}
    with open(args.annotation_file) as f:
        annotation_dict = json.loads(f.read())

    if args.mode == 1:
        samples = gen_samples(annotation_dict, label_id_dict,
                         args.fps, args.feats_dir,
                         args.train_postfix,
                         args.extract_youtube8m, args.extract_stft)
        random.shuffle(samples)
        val_len = int(len(samples) * args.ratio)
        train_len = len(samples) - val_len
        with open(args.samples_dir + '/train', 'w') as scene_train_fs:
            write_samples(samples[:train_len], scene_train_fs)
        with open(args.samples_dir + '/val', 'w') as scene_val_fs:
            write_samples(samples[train_len:], scene_val_fs)
    elif args.mode == 2:
        samples = gen_samples(annotation_dict, label_id_dict,
                         args.fps, args.feats_dir,
                         args.test_postfix,
                         args.extract_youtube8m, args.extract_stft)
        with open(args.samples_dir + '/test', 'w') as scene_test_fs:
            write_samples(samples, scene_test_fs)


