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

def _inference(args, model, data_youtube8m, data_stft, threshold, video_id, index, ts, label):
    if args.use_gpu == 1:
        try:
            data_youtube8m = data_youtube8m.cuda()
            data_stft = data_stft.cuda()
        except Exception as e:
            print(e)
            return []
    output = model(data_youtube8m, data_stft)
    output = output.view(-1, 2)
    output = functional.softmax(output, dim=1)
    probs = output[:, 1].cpu().detach().numpy()
    predicts = np.nan_to_num(probs > threshold).tolist()
    probs = probs.tolist()
    res = []
    for i in range(len(label)):
        res.append([video_id[i], index[i], ts[i], probs[i], predicts[i], label[i]])
    return res

def inference(args, model, data_loader, threshold):
    model.eval()
    res = []
    with torch.no_grad():
        with ThreadPoolExecutor(max_workers=args.max_worker) as executor:
            for ps in gen_batch(data_loader, executor, args, threshold):
                for p in ps:
                    t = p.result()
                    res.extend(t)

    return res

def gen_batch(data_loader, executor, args, threshold):
    cnt = 0
    ps = []
    for data_youtube8m, data_stft, label, video_id, index, ts in tqdm.tqdm(data_loader, total = len(data_loader)):
        cnt += 1
        ps.append(executor.submit(_inference, args, model,
            data_youtube8m, data_stft, threshold,
            video_id, index.numpy().tolist(), ts.numpy().tolist(), label.numpy().tolist()))
        if cnt % args.max_worker == 0:
            yield ps
            ps = []
    if cnt > 0:
        yield ps

def save(res, path):
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
            predict = x[4]
            label = x[5]
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
        if pre_video_id != '':
            obj[pre_video_id]['annotations'].append({'segment': [s, e], 'labels': []})
        json.dump(obj, fs)

def val(res, threshold):
    probs = [x[3] for x in res]
    predicts = [1 if x[3] > threshold else 0 for x in res]
    labels = [x[5] for x in res]
    auc = sklearn.metrics.roc_auc_score(labels, probs)
    acc = sklearn.metrics.accuracy_score(labels, predicts)
    recall = sklearn.metrics.recall_score(labels, predicts)
    precision = sklearn.metrics.precision_score(labels, predicts)
    ap = sklearn.metrics.average_precision_score(labels, probs)
    f1 = sklearn.metrics.f1_score(labels, predicts)
    print('threshold: {}, auc: {}, acc: {}, recall: {}, precision: {}, ap: {}, f1: {}'.format(threshold, auc, acc, recall, precision, ap, f1))

def load_model(args):
    model = LGSS(args)
    if args.use_gpu == 1:
        model = model.cuda()
    model = nn.DataParallel(model)
    #checkpoint = load_checkpoint(os.path.join(args.model_dir, 'model_best.pth.tar'), args)
    checkpoint = load_checkpoint(os.path.join(args.model_dir, 'checkpoint.pth.tar'), args)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def run(args, model):
    test_samples_path = os.path.join(args.samples_dir, args.test_postfix)
    val_samples_path = os.path.join(args.samples_dir, args.train_postfix)

    val_loader = DataLoader(
        SegDataset(val_samples_path, args.extract_youtube8m, args.extract_stft, 'train_5k_A', args.feats_dir, False),
        num_workers=5,
        prefetch_factor=100,
        batch_size=20,
        shuffle=False)

    test_loader = DataLoader(
        SegDataset(test_samples_path, args.extract_youtube8m, args.extract_stft, 'test_5k_A', args.feats_dir, False),
        num_workers=5,
        prefetch_factor=100,
        batch_size=20,
        shuffle=False)

    '''
    print('start val...')
    res = inference(args, model, val_loader, 0.55)
    save(res, os.path.join(args.result_dir, 'val'))
    for threshold in np.arange(0, 1.05, 0.05).tolist():
        val(res, threshold)
    '''

    print('start inference...')
    res = inference(args, model, test_loader, 0.7)
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
