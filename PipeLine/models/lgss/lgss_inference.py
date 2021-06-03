import torch
import os
from torch.utils.data import DataLoader
from dataset.seg_dataset import SegDataset
import argparse
from models.lgss.lgss import LGSS
import torch.nn as nn
import glob
from utils.torch_utils import *
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import torch.nn.functional as functional
import numpy as np
import tqdm
import sklearn

def inference(args, model, data_loader, threshold):
    model.eval()
    res = []
    with torch.no_grad():
        for data_youtube8m, data_stft, label, video_id, index, ts in tqdm.tqdm(data_loader, total = len(data_loader), desc='inference'):
            if args.use_gpu == 1:
                data_youtube8m = data_youtube8m.cuda()
                data_stft = data_stft.cuda()

            output = model(data_youtube8m, data_stft)
            output = output.view(-1, 2)

            output = functional.softmax(output, dim=1)
            probs = output[:, 1].cpu().detach().numpy()
            predicts = np.nan_to_num(probs > threshold).tolist()
            probs = probs.tolist()
            res.append([video_id[0], index[0], ts[0], probs[0], predicts[0], label.cpu().detach().numpy().tolist()[0]])
    return res

def save(res, path):
    with open(path, 'w') as fs:
        for x in res:
            video_id = x[0]
            index = x[1]
            ts = x[2]
            prob = x[3]
            predict = x[4]
            label = x[5]
            fs.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(video_id, index, ts, prob, predict, label))

def val(res):
    probs = [x[3] for x in res]
    predicts = [x[4] for x in res]
    labels = [x[5] for x in res]
    auc = sklearn.metrics.roc_auc_score(labels, probs)
    acc = sklearn.metrics.accuracy_score(labels, predicts)
    recall = sklearn.metrics.recall_score(labels, predicts)
    precision = sklearn.metrics.precision_score(labels, predicts)
    ap = sklearn.metrics.average_precision_score(labels, probs)
    f1 = sklearn.metrics.f1_score(labels, predicts)
    print('auc: {}, acc: {}, recall: {}, precision: {}, ap: {}, f1: {}'.format(auc, acc, recall, precision, ap, f1))

def load_model(args):
    model = LGSS(args)
    if args.use_gpu == 1:
        model = model.cuda()
    model = nn.DataParallel(model)
    checkpoint = load_checkpoint(os.path.join(args.model_dir, 'model_best.pth.tar'), args.use_gpu)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def run(args, model, video_name):
    test_samples_path = os.path.join(args.samples_dir, 'test')
    val_samples_path = os.path.join(args.samples_dir, 'val')

    val_loader = DataLoader(
        SegDataset(test_samples_path, args.extract_youtube8m, args.extract_stft, 'train_5k_A', args.feats_dir, False),
        num_workers=10,
        prefetch_factor=100,
        batch_size=1,
        shuffle=True)

    test_loader = DataLoader(
        SegDataset(val_samples_path, args.extract_youtube8m, args.extract_stft, 'test_5k_A', args.feats_dir, False),
        num_workers=10,
        prefetch_factor=100,
        batch_size=1,
        shuffle=False)

    res = inference(args, model, val_loader)
    save(res, os.path.join(args.result_dir, 'val'))
    val(res)

    res = inference(args, model, test_loader)
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

    args = parser.parse_args()
    os.makedirs(args.result_dir)
    if args.use_gpu == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

    print('load model')
    model = load_model(args)

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for video_path in glob.glob(os.path.join(args.video_dir, '*.mp4')):
            video_name = os.path.basename(video_path).split(".m")[0]
            results.append(executor.submit(args, run, model, video_name))
        results = [res.result() for res in results]
    print('done')