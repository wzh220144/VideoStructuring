from __future__ import print_function, absolute_import
import argparse
import os
import os.path as osp
from collections import OrderedDict
from datetime import datetime
import multiprocessing
import numpy as np
import pickle
import pdb
from PIL import Image
import sys
import json
import time
from pprint import pprint
import cv2
import tqdm
import torch
import torch.nn as nn

from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTModel

class VitExtractor(object):
  def __init__(self, args):
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch32-384')
        self.model = ViTModel.from_pretrained('google/vit-base-patch32-384')
        if args.use_gpu == 1:
            self.model = nn.DataParallel(self.model.cuda().to('cuda:0'))

  def extract_rgb_frame_features_list(self, frame_rgbs, count):
        inputs = self.feature_extractor(images=frame_rgbs, return_tensors="pt")
        if args.use_gpu == 1:
            inputs = {k: v.cuda().to('cuda:0') for k, v in inputs.items()}
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        return last_hidden_state[:,0,:].cpu().detach().numpy()

def gen_batch(video_dir_list, args):
    count = 0
    rgbs = []
    feat_paths = []
    for video_dir in tqdm.tqdm(video_dir_list):
        img_paths = sorted(os.listdir(video_dir))
        video_id = video_dir.split('/')[-1]
        os.makedirs(args.feat_path + '/' + video_id, exist_ok=True)
        for img_path in img_paths:
            cols = img_path.split('_')
            img_id = cols[3].split('.')[0]
            shot_id = cols[0] + '_' + cols[1]
            feat_path = args.feat_path + '/' + video_id + '/' + shot_id + '.npy'
            if os.path.exists(feat_path):
                continue
            if int(img_id) == args.keyf_num:
                img = cv2.imread(video_dir + '/' + img_path)
                img = img[:,:,::-1].transpose((2,0,1))

                rgbs.append(img)
                feat_paths.append(feat_path)
                count += 1
                if count == args.batch_size:
                    yield rgbs, count, feat_paths
                    count = 0
                    rgbs = []
                    feat_paths = []
    if count > 0:
        yield rgbs, count, feat_paths

def main(args):
    os.makedirs(args.feat_path, exist_ok=True)
    if args.use_gpu == 1:
        os.environ["CUDA_VISIBLE_DEVICES"]='0,1'
    extractor = VitExtractor(args)
    
    video_dir_list = [args.source_img_path + '/' + x for x in sorted(os.listdir(args.source_img_path))]
    print('****** Total {} videos ******'.format(len(video_dir_list)))

    for frame_rgbs, count, feat_paths in gen_batch(video_dir_list, args):
        features = extractor.extract_rgb_frame_features_list(frame_rgbs, count)
        for index in range(count):
            features[index].dump(feat_paths[index])
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', type=int, default=1)
    parser.add_argument('--data_root', type=str, default="/home/tione/notebook/dataset/train_5k_A/shot_hsv")
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--keyf_num', type=int, default=0)
    args = parser.parse_args()
    args.list_file = None
    args.source_img_path = osp.join(args.data_root, 'shot_keyf')
    args.feat_path = osp.join(args.data_root, 'vit_feat')
    main(args)
