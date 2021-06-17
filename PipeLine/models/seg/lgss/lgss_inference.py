from torch.utils.data import DataLoader
from dataset.seg_dataset import SegDataset
import argparse
from models.seg.lgss.lgss import LGSS
import torch.nn as nn
import glob
from utils.torch_utils import *
from concurrent.futures import ThreadPoolExecutor
import torch.nn.functional as functional
import numpy as np
import tqdm
import sklearn
import sklearn.metrics
import json
import models.seg.lgss.lgss_util as lgss_util

def save(res, path, threshold):
    obj = {}
    res = sorted(res, key=lambda x: (x[0], x[1]))
    with open(path, 'w') as fs:
        s = 0
        e = 0
        pre_video_id = ''
        for x in res:
            video_id = x[0] + '.mp4'
            index = x[1]
            ts = x[2]
            prob = x[3]
            label = x[4]
            predict = 0
            if prob > threshold:
                predict = 1
            if video_id not in obj:
                if pre_video_id != '':
                    obj[pre_video_id]['annotations'].append({'segment': [s, e], 'labels': []})
                obj[video_id] = {'annotations': []}
                s = 0
                e = 0
            e = ts
            if predict == 1:
                obj[video_id]['annotations'].append({'segment': [s, e], 'labels': []})
                s = e
            pre_video_id = video_id
        if pre_video_id != '' and s != e:
            obj[pre_video_id]['annotations'].append({'segment': [s, e], 'labels': []})
        json.dump(obj, fs)

def load_model(args):
    model = LGSS(args)
    if args.use_gpu == 1:
        model = model.cuda()
    model = nn.DataParallel(model)
    checkpoint = load_checkpoint(os.path.join(args.model_dir, 'model_best.pth.tar'), args)
    #checkpoint = load_checkpoint(os.path.join(args.model_dir, 'checkpoint.pth.tar'), args)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def run(args, model):
    test_samples_path = os.path.join(args.samples_dir, args.test_postfix)
    val_samples_path = os.path.join(args.samples_dir, args.train_postfix)

    '''
    val_loader = DataLoader(
        SegDataset(val_samples_path, args.extract_youtube8m, args.extract_stft, 'train_5k_A', args.feats_dir, False),
        num_workers=5,
        prefetch_factor=100,
        batch_size=20,
        shuffle=False)
    '''

    test_loader = DataLoader(
        SegDataset(test_samples_path, args.extract_youtube8m, args.extract_stft, 'test_5k_A', args.feats_dir, False),
        num_workers=5,
        prefetch_factor=100,
        batch_size=20,
        shuffle=False)

    print('start inference...')
    criterion = nn.CrossEntropyLoss(torch.Tensor([0.5, 5]))
    if args.use_gpu == 1:
        criterion = criterion.cuda()
    res, total_loss = lgss_util.inference(args, model, test_loader, criterion)
    save(res, os.path.join(args.result_dir, 'test'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', type=int, default=1)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--samples_dir', type=str, default='/home/tione/notebook/VideoStructuring/dataset/samples/seg')
    parser.add_argument('--model_dir', type=str, default='/home/tione/notebook/VideoStructuring/dataset/model/seg')
    parser.add_argument('--feats_dir', type=str, default='/home/tione/notebook/VideoStructuring/dataset/feats')
    parser.add_argument('--youtube8m_cache_size', type=int, default=10000)
    parser.add_argument('--stft_cache_size', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--extract_youtube8m', type=bool, default=True)
    parser.add_argument('--extract_stft', type=bool, default=True)
    parser.add_argument('--youtube8m_ratio', type=float, default=0.8)
    parser.add_argument('--stft_ratio', type=float, default=0.2)
    parser.add_argument('--sim_dim', type=int, default=512)
    parser.add_argument('--youtube8m_dim', type=int, default=1024)
    parser.add_argument('--stft_dim', type=int, default=512)
    parser.add_argument('--lstm_hidden_size', type=int, default=512)
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--bidirectional', type=bool, default=True)
    parser.add_argument('--stft_feat_dim', type=int, default=512)
    parser.add_argument('--logs_dir', type=str, default='/home/tione/notebook/VideoStructuring/dataset/log/seg')
    parser.add_argument('--result_dir', type=str, default='/home/tione/notebook/VideoStructuring/dataset/result/seg')
    parser.add_argument('--max_worker', type=int, default=10)
    parser.add_argument('--train_postfix', type=str, default='train_5k_A')
    parser.add_argument('--test_postfix', type=str, default='test_5k_A')

    args = parser.parse_args()
    os.makedirs(args.result_dir, exist_ok=True)
    if args.use_gpu == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

    print('load model')
    model = load_model(args)
    run(args, model)
