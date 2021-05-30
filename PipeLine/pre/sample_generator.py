#coding=utf-8

import os
import argparse
import json
import cv2
import tqdm
import random

def read_video_info(cap):
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    return frame_count, fps, h, w

def parse_annotation(train_txt, video_dir):
    with open(train_txt) as f:
        annotation_dict = json.loads(f.read())
    scene_list = []
    tag_list = []
    for key, value in tqdm.tqdm(annotation_dict.items()):
        video_id = key.strip().split('.mp4')[0]
        video_path = os.path.join(video_dir, key)
        cap = cv2.VideoCapture(video_path)
        frame_count, fps, h, w = read_video_info(cap)
        annotations = value["annotations"]
        if annotations[0]['segment'][0] != 0:
            print('annotation of {} start time error.'.format(video_id))
            continue
        flag = True
        annotation_index = 0
        frame_index = 0
        t1 = []
        t2 = []
        for annotation in annotations:
            segment = annotation['segment']
            labels = annotation['labels']
            start_ts = segment[0]
            end_ts = segment[1]
            if start_ts > end_ts:
                flag = False
                break
            status = 0
            start_frame = frame_index + 1
            while True:
                has_frame, frame = cap.read()
                if not has_frame:
                    break
                cur_ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                scene_label = 0
                if cur_ts >= end_ts:
                    scene_label = 1
                    status = 1
                t1.append([video_id, annotation_index, frame_index, scene_label, labels, frame_count, fps, h, w])
                frame_index += 1
                if status == 1:
                    break
            end_frame = frame_index
            t2.append(video_id, annotation_index, start_frame, end_frame, labels, frame_count, fps, h, w)
            annotation_index += 1
        if flag:
            print('annotation of {} segment error.'.format(video_id))
            continue
        t1.pop()  #discard last frame
        scene_list.extend(t1)
        tag_list.extend(t2)
        cap.release()
    return scene_list, tag_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type = str, default = "/home/tione/notebook/VideoStructuring/dataset/videos/train_5k_A")
    parser.add_argument('--data_root', type = str, default = "/home/tione/notebook/VideoStructuring/dataset")
    parser.add_argument('--train_txt', type = str, default = "/home/tione/notebook/VideoStructuring/dataset/structuring/GroundTruth/train5k.txt")
    parser.add_argument('ratio', type = float, default=0.05)
    args = parser.parse_args()
    print(args)

    scene_train_fs = open(args.data_root + '/scene_train', 'w')
    scene_val_fs = open(args.data_root + '/scene_val', 'w')
    tag_train_fs = open(args.data_root + '/tag_train', 'w')
    tag_val_fs = open(args.data_root + '/tag_val', 'w')
    scene_list, tag_list = parse_annotation(args.train_txt, args.train_dir)
    scene_list = random.shuffle(scene_list)
    tag_list = random.shuffle(tag_list)
    for x in scene_list:
        p = random.random()
        if p < args.ratio:
            scene_val_fs.write('\t'.join([str(xx)for xx in x]) + '\n')
        else:
            scene_train_fs.write('\t'.join([str(xx) for xx in x]) + '\n')
    for x in tag_list:
        p = random.random()
        if p < args.ratio:
            tag_val_fs.write('\t'.join([str(xx)for xx in x]) + '\n')
        else:
            tag_train_fs.write('\t'.join([str(xx) for xx in x]) + '\n')