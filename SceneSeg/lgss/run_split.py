from __future__ import print_function

from mmcv import Config
import lgss.models as models
import torch
import torch.nn as nn
from lgss.data.get_data import get_inference_data
from torch.utils.data import DataLoader
import lgss.utils
from lgss.utils import (load_checkpoint, scene2video)
from utils.package import *
import glob
from concurrent.futures import ThreadPoolExecutor
import json

final_dict = {}



def main(cfg, args, video_name, value, threshold):
    end_frames = sorted([int(key) for key in value.keys()])
    start_frame = 0
    scene_list = []
    for end_frame in end_frames:
        v = value[str(end_frame)]
        label = 0
        if v['prob'] >= threshold:
            label = 1
        if label == 1:
            scene_list.append([start_frame, end_frame])
            start_frame = end_frame + 1
    topn = args.topn
    smooth_threshold = threshold - args.smooth_threshold
    if len(scene_list) == 0:
        t = sorted([(int(k), v) for k, v in value.items()], key=lambda x: -x[1]['prob'])
        frames = []
        for index in range(topn):
            if index < len(t) and t[index][1]['prob'] >= smooth_threshold:
                frames.append(t[index][0])
        start_frame = 0
        frames = sorted(frames)
        for frame in frames:
            scene_list.append([start_frame, frame])
            start_frame = frame + 1
    scene2video(cfg, scene_list, args, video_name)

def parse_args():
    parser = argparse.ArgumentParser(description='Runner')
    parser.add_argument('--config', help='config file path', default = '/home/tione/notebook/VideoStructuring/SceneSeg/lgss/config/inference_hsv.py')
    parser.add_argument('--max_workers', type=int, default = 30)
    parser.add_argument('--use_gpu', type=int, default = 1)
    parser.add_argument('--threshold', type=float, default=0.94)
    parser.add_argument('--topn', type=int, default=-1)
    parser.add_argument('--smooth_threshold', type=float, default=0.1)
    parser.add_argument('--split_file', type=str, default='/home/tione/notebook/dataset/train_5k_A/shot_transnet_v2/meta/split.json')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus
    assert cfg.testFlag, "testFlag must be True"

    test = set()
    with open(args.split_file, 'r') as f:
        obj = json.load(f)
        test = set(obj['val'])

    video_inference_res = {}
    for video_path in glob.glob(os.path.join(cfg.video_dir, '*.mp4')):
        video_name = os.path.basename(video_path).split(".m")[0]
        if video_name not in test:
            continue
        video_inference_res[video_name] = {}
        log_path = os.path.join(cfg.data_root, "seg_results", video_name + '.json')
        if osp.exists(log_path):
            #print(log_path + ' exist.')
            with open(log_path, 'r') as f:
                video_inference_res[video_name] = json.load(f)
            continue
        else:
            print(log_path + ' does not exist.')
            video_inference_res[video_name] = {}

    os.makedirs(cfg.output_root, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers = args.max_workers) as executor:
        for video_name, value in video_inference_res.items():
            results.append(executor.submit(main, cfg, args, video_name, value, args.threshold))
        for res in results:
            res.result()
