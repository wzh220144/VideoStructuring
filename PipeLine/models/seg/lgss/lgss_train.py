from torch.utils.data import DataLoader
from dataset.seg_dataset import SegDataset
import dataset.seg_dataset as seg_dataset
import argparse
from models.seg.lgss.lgss import LGSS
import torch.nn as nn
from utils.torch_utils import *
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import MultiStepLR
from tensorboardX import SummaryWriter
import torch.nn.functional as functional
import numpy as np
import tqdm
import sklearn
import sklearn.metrics
from concurrent.futures import ThreadPoolExecutor
import json
import models.seg.lgss.lgss_util as lgss_util

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

train_iter = 0
val_iter = 0

annotation_dict = {}

def train(args, model, data_loader, optimizer, scheduler, epoch, criterion, val_loader, writer, best_f1, best_threshold):
    global train_iter
    model.train()
    labels = []
    preds = []
    probs = []
    total_loss = 0.0
    cnt = 0
    for batch_idx, (youtube8m_data, stft_data, label, _, _, _) in enumerate(tqdm.tqdm(data_loader, total = len(data_loader))):
        label = label.view(-1)
        if args.use_gpu == 1:
            youtube8m_data = youtube8m_data.cuda()
            stft_data = stft_data.cuda()
            label = label.cuda()
        optimizer.zero_grad()
        try:
            output = model(youtube8m_data, stft_data)
        except Exception as e:
            print(e)
            continue
        output = output.view(-1, 2)
        loss = criterion(output, label)
        loss.backward()
        prob = functional.softmax(output, dim=1)[:, 1].cpu().detach().numpy()
        pred = np.nan_to_num(prob) > 0.5
        optimizer.step()
        train_iter += 1

        _label = label.cpu().detach().numpy().tolist()
        _pred = pred.tolist()
        _prob = prob.tolist()
        preds.extend(_pred)
        probs.extend(_prob)
        labels.extend(_label)
        total_loss += loss.item()
        cnt += 1

        if cnt % 100 == 0:
            labels = []
            preds = []
            probs = []
            total_loss = 0
            cnt = 0
            acc = sklearn.metrics.accuracy_score(labels, preds)
            recall = sklearn.metrics.recall_score(labels, preds, zero_division=1)
            precision = sklearn.metrics.precision_score(labels, preds, zero_division=1)
            auc = sklearn.metrics.roc_auc_score(labels, probs)
            ap = sklearn.metrics.average_precision_score(labels, probs)
            f1 = sklearn.metrics.f1_score(labels, probs, zero_division=1)

            writer.add_scalar('train/loss', total_loss / cnt, train_iter)
            writer.add_scalar('train/auc', auc, train_iter)
            writer.add_scalar('train/acc', acc, train_iter)
            writer.add_scalar('train/recall', recall, train_iter)
            writer.add_scalar('train/precision', precision, train_iter)
            writer.add_scalar('train/ap', ap, train_iter)
            writer.add_scalar('train/f1', f1, train_iter)

            t = 'epoch {}: [{}/{} ({:.0f}%)]\t' + \
                'loss: {:.6f}, auc: {:.6f}, acc: {:.6f}, recall: {:.6f}, precision: {:.6f}, ap: {:.6f}, f1: {:.6f}'
            print(t.format(
                epoch,
                int(batch_idx * len(youtube8m_data)),
                len(data_loader.dataset),
                100. * batch_idx / len(data_loader),
                total_loss / cnt,
                auc,
                acc,
                recall,
                precision,
                ap,
                f1,
                ))
        if batch_idx % 10000 == 0 and batch_idx != 0:
            best_f1, best_threshold = test(args, model, val_loader, best_f1, best_threshold, criterion, epoch)

        scheduler.step()
    return best_f1, best_threshold

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', type = int, default = 1)
    parser.add_argument('--resume', type = str, default = None)
    parser.add_argument('--samples_dir', type = str, default = '/home/tione/notebook/VideoStructuring/dataset/samples/seg')
    parser.add_argument('--model_dir', type = str, default = '/home/tione/notebook/VideoStructuring/dataset/model/seg')
    parser.add_argument('--feats_dir', type = str, default = '/home/tione/notebook/VideoStructuring/dataset/feats')
    parser.add_argument('--youtube8m_cache_size', type = int, default = 10000)
    parser.add_argument('--stft_cache_size', type = int, default = 10000)
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--extract_youtube8m', type = bool, default = True)
    parser.add_argument('--extract_stft', type = bool, default = True)
    parser.add_argument('--youtube8m_ratio', type = float, default = 0.8)
    parser.add_argument('--stft_ratio', type = float, default = 0.2)
    parser.add_argument('--sim_dim', type = int, default = 512)
    parser.add_argument('--youtube8m_dim', type = int, default = 1024)
    parser.add_argument('--stft_dim', type = int, default = 512)
    parser.add_argument('--lstm_hidden_size', type = int, default = 512)
    parser.add_argument('--window_size', type = int, default = 5)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--bidirectional', type=bool, default=True)
    parser.add_argument('--stft_feat_dim', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--logs_dir', type=str, default='/home/tione/notebook/VideoStructuring/dataset/log/seg')
    parser.add_argument('--max_worker', type=int, default=5)
    parser.add_argument('--train_postfix', type = str, default = 'train_5k_A')
    parser.add_argument('--result_dir', type=str, default='/home/tione/notebook/VideoStructuring/dataset/result/seg')
    parser.add_argument('--annotation_file', type=str, default='/home/tione/notebook/dataset/structuring/GroundTruth/train5k.txt')

    args = parser.parse_args()

    global annotation_dict
    annotation_dict = {}
    with open('{}/{}'.format(args.result_dir, args.train_postfix)) as f:
        annotation_dict = json.loads(f.read())

    seg_dataset.youtube8m_cache_size = args.youtube8m_cache_size
    seg_dataset.stft_cache_size = args.stft_cache_size

    train_samples_path = os.path.join(args.samples_dir, 'train_5k_A')
    val_samples_path = os.path.join(args.samples_dir, 'val_train_5k_A')

    train_loader = DataLoader(
            SegDataset(train_samples_path, args.extract_youtube8m, args.extract_stft, 'train_5k_A', args.feats_dir, False), 
            num_workers = 5,
            prefetch_factor = 2,
            batch_size=args.batch_size, 
            shuffle=True)
    val_loader = DataLoader(
            SegDataset(val_samples_path, args.extract_youtube8m, args.extract_stft, 'train_5k_A', args.feats_dir, False), 
            num_workers = 5,
            prefetch_factor = 2,
            batch_size=20, shuffle=False)

    model = LGSS(args)
    if args.use_gpu == 1:
        model = model.cuda()
    model = nn.DataParallel(model)
    if args.resume is not None:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])

    optimizer = Adam(model.parameters(), lr = 1e-3, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones = [5000, 10000, 30000])
    criterion = nn.CrossEntropyLoss(torch.Tensor([0.5, 5]))
    if args.use_gpu == 1:
        criterion = criterion.cuda()

    print("start training...")
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)

    writer = SummaryWriter(logdir=args.logs_dir)
    best_f1 = -1
    best_threshold = -1
    for epoch in range(args.epochs):
        best_f1, best_threshold = train(args, model, train_loader, optimizer, scheduler, epoch, criterion, val_loader, writer, best_f1, best_threshold)
        best_f1, best_threshold = test(args, model, val_loader, best_f1, best_threshold, criterion, epoch)

def test(args, model, val_loader, best_f1, best_threshold, criterion, epoch):
    print('start val...')
    res, total_loss = lgss_util.inference(args, model, val_loader, criterion)

    cur_max_threshold = -1
    cur_max_f1 = 0
    is_best = False
    for threshold in np.arange(0, 1.01, 0.01).tolist():
        auc, acc, recall, precision, ap, f1, avg_loss, f1_w = lgss_util.val(res, threshold, total_loss, args)
        print('threshold: {}, auc: {}, acc: {}, recall: {}, precision: {}, ap: {}, f1: {}, avg_loss: {}, f1_w: {}'.format(threshold, auc, acc, recall, precision, ap, f1, avg_loss, f1_w))
        if f1_w > best_f1:
            is_best = True
            best_f1 = f1_w
            best_threshold = threshold
        if f1_w > cur_max_f1:
            cur_max_f1 = f1_w
            cur_max_threshold = threshold
    print('epoch {}: \tcur_max_threshold:{:.6f}, best_threshold: {:.6f}, cur_max_f1: {:.6f}, best_f1: {:.6f}'.format(
        epoch, cur_max_threshold, best_threshold,
        cur_max_f1, best_f1))
    
    save_checkpoint({'state_dict': model.state_dict(), 'epoch': epoch + 1,}, 
            is_best = is_best, fpath = osp.join(args.model_dir, 'checkpoint.pth.tar'))
    return best_f1, best_threshold

if __name__ == '__main__':
    main()
