#coding=utf-8

import os
import argparse
import json
import cv2
import tqdm
import random
from utils import utils
import glob

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

def do_gen_samples(info, window_size):
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

def gen_samples(annotation_dict, label_id_dict, fps, window_size, feats_dir, video_dir, postfix, extract_youtube8m, extract_stft):
    youtube8m_feats_dir = os.path.join(feats_dir, postfix, 'youtube8m')
    stft_feats_dir = os.path.join(feats_dir, postfix, 'stft')
    res = []
    for video_file in glob.glob(os.path.join(args.video_dir, postfix, '*.mp4')):
        print(video_file)
        info = read_video_info(video_file, fps, youtube8m_feats_dir, stft_feats_dir, extract_youtube8m, extract_stft)
        #print(info)
        video_name = video_file.split('/')[-1]
        video_annotation = {}
        if video_name in annotation_dict:
            video_annotation = annotation_dict[video_name]
        info, flag = gen_labels(info, label_id_dict, video_annotation)
        #print(info)
        if not flag:
            continue
        samples = do_gen_samples(info, window_size)
        #print(samples)
        res.extend(samples)
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
    parser.add_argument('--train_txt', type = str, default = "/home/tione/notebook/VideoStructuring/dataset/structuring/GroundTruth/train5k.txt")
    parser.add_argument('--label_id', type = str, default = '/home/tione/notebook/VideoStructuring/dataset/label_id.txt')
    parser.add_argument('--feats_dir', type = str, default = '/home/tione/notebook/VideoStructuring/dataset/feats')
    parser.add_argument('--extract_youtube8m', type = bool, default = True)
    parser.add_argument('--extract_stft', type = bool, default = True)
    parser.add_argument('--fps', type = int, default = 5)
    parser.add_argument('--ratio', type = float, default = 0.05)
    parser.add_argument('--samples_dir', type = str, default = '/home/tione/notebook/VideoStructuring/dataset/samples/seg')
    parser.add_argument('--window_size', type = int, default = 5)
    args = parser.parse_args()

    os.makedirs(args.samples_dir, exist_ok=True)
    label_id_dict = parse_label_id(args.label_id)
    annotation_dict = {}
    with open(args.train_txt) as f:
        annotation_dict = json.loads(f.read())
    samples = gen_samples(annotation_dict, label_id_dict,
                     args.fps, args.window_size, args.feats_dir,
                     args.video_dir, args.train_postfix,
                     args.extract_youtube8m, args.extract_stft)
    random.random(samples)
    val_len = int(len(samples) * args.ratio)
    train_len = len(samples) - val_len
    with open(args.samples_dir + '/train', 'w') as scene_train_fs:
        write_samples(samples[:train_len], scene_train_fs)
    with open(args.samples_dir + '/val', 'w') as scene_val_fs:
        write_samples(samples[train_len:], scene_val_fs)

    '''
    samples = gen_samples({}, label_id_dict, 
                     args.fps, args.seq_len, args.feats_dir,
                     args.video_dir, args.test_postfix,
                     args.extract_youtube8m, args.extract_stft)
    with open(args.samples_dir + '/test', 'w') as scene_test_fs:
        write_samples(samples, scene_test_fs)
    '''

