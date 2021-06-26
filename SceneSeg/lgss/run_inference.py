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
import lgss.models.lgss_util as lgss_util
import lgss.models.lgss as lgss

final_dict = {}

def load_model(cfg, args, use_best = True):
    model = lgss.LGSS(cfg)
    if args.use_gpu == 1:
        model = model.cuda()
    if use_best:
        checkpoint = load_checkpoint(cfg.model_path + '/model_best.pth.tar', args.use_gpu)
    else:
        checkpoint = load_checkpoint(cfg.model_path + '/checkpoint.pth.tar', args.use_gpu)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    threshold = checkpoint['checkpoint']
    if args.use_gpu == 1:
        model = nn.DataParallel(model)
    return model, threshold

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
    if end_frames[-1] > start_frame:
        scene_list.append([start_frame, end_frames[-1]])
    scene2video(cfg, scene_list, args, video_name)

def parse_args():
    parser = argparse.ArgumentParser(description='Runner')
    parser.add_argument('--config', help='config file path', default = '/home/tione/notebook/VideoStructuring/SceneSeg/lgss/config/inference_hsv.py')
    parser.add_argument('--max_workers', type=int, default = 30)
    parser.add_argument('--use_gpu', type=int, default = 1)
    parser.add_argument('--threshold', type=float, default=0.93)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus
    assert cfg.testFlag, "testFlag must be True"

    model, threshold = load_model(cfg, args)
    print('model complete')

    video_names = []
    video_inference_res = {}
    for video_path in glob.glob(os.path.join(cfg.video_dir, '*.mp4')):
        video_name = os.path.basename(video_path).split(".m")[0]
        video_inference_res[video_name] = {}
        log_path = os.path.join(cfg.data_root, "seg_results", video_name + '.json')
        if osp.exists(log_path):
            #print(log_path + ' exist.')
            with open(log_path, 'r') as f:
                video_inference_res[video_name] = json.load(f)
            continue
        #print(log_path + ' does not exist.')
        video_names.append(video_name)
    data = get_inference_data(cfg, video_names)
    data_loader = DataLoader(data, batch_size=cfg.batch_size, shuffle=False, **cfg.data_loader_kwargs)

    criterion = nn.CrossEntropyLoss(torch.Tensor(cfg.loss.weight).cuda())
    inference_res, total_loss = lgss_util.inference(cfg, args, model, data_loader, criterion)
    for x in inference_res:
        label = x[0]
        prob = x[1]
        end_frame = x[2]
        video_name = x[3]
        if video_name not in video_inference_res:
            video_inference_res[video_name] = {}
        k = str(end_frame)
        if k not in video_inference_res[video_name]:
            video_inference_res[video_name][k] = {'prob': prob.item(), 'label': label}
        video_inference_res[video_name][k]['prob'] = max(video_inference_res[video_name][k]['prob'], prob.item())
    os.makedirs(os.path.join(cfg.data_root, "seg_results"), exist_ok=True)
    for video_name, value in video_inference_res.items():
        log_path = os.path.join(cfg.data_root, "seg_results", video_name + '.json')
        with open(log_path, 'w') as f:
            json.dump(value, f, ensure_ascii=False, indent=4)

    results = []
    with ThreadPoolExecutor(max_workers = args.max_workers) as executor:
        for video_name, value in video_inference_res.items():
            results.append(executor.submit(main, cfg, args, video_name, value, threshold))
        for res in results:
            res.result()
    print('done')
