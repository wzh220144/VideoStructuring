import torch
import os
from torch.utils.data import DataLoader
from dataset.seg_dataset import SegDataset
import dataset.seg_dataset as seg_dataset
import argparse
from models.lgss import LGSS
import torch.nn as nn
from utils.torch_utils import *
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import MultiStepLR
from tensorboardX import SummaryWriter
import torch.nn.functional as functional
import numpy as np
from sklearn.metrics import average_precision_score

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

train_iter = 0
val_iter = 0

def get_ap(gts_raw, preds_raw):
    gts, preds = [], []
    for gt_raw in gts_raw:
        gts.extend(gt_raw.tolist())
    for pred_raw in preds_raw:
        preds.extend(pred_raw.tolist())
    return average_precision_score(np.nan_to_num(gts), np.nan_to_num(preds))

def get_mAP_seq(loader, gts_raw, preds_raw):
    mAP = []
    gts, preds = [], []
    for gt_raw in gts_raw:
        gts.extend(gt_raw.tolist())
    for pred_raw in preds_raw:
        preds.extend(pred_raw.tolist())

    seq_len = len(loader.dataset.listIDs[0])
    n = min(len(loader.dataset), len(gts) // seq_len, len(preds) // seq_len)
    lines = []
    for i in range(n):
        for j in range(seq_len):
            one_idx = loader.dataset.listIDs[i][j]
            line = '{} {} {} {}'.format(one_idx['imdbid'], one_idx['shotid'], \
                                        gts[i * seq_len + j], preds[i * seq_len + j])
            lines.append(line)
    lines = list(sorted(lines))
    imdbs = np.array([x.split(' ')[0] for x in lines])
    # shots = np.array([x.split(' ')[1] for x in lines])
    gts = np.array([int(x.split(' ')[2]) for x in lines], np.int32)
    preds = np.array([float(x.split(' ')[3]) for x in lines], np.float32)
    movies = np.unique(imdbs)
    # print("movies:", movies)
    for movie in movies:
        index = np.where(imdbs == movie)[0]
        ap = average_precision_score(np.nan_to_num(gts[index]), np.nan_to_num(preds[index]))
        mAP.append(round(np.nan_to_num(ap), 2))
        # print(mAP)
    return np.mean(mAP), np.array(mAP)

def train(args, model, data_loader, optimizer, scheduler, epoch, criterion, writer):
    global train_iter
    model.train()
    for batch_idx, (youtube8m_data, stft_data, label) in enumerate(data_loader):
        label = label.view(-1)
        if args.use_gpu == 1:
            youtube8m_data = youtube8m_data.cuda()
            stft_data = stft_data.cuda()
            label = label.cuda()
        optimizer.zero_grad()
        output = model(youtube8m_data, stft_data)
        output = output.view(-1, 2)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        if loss.item() > 0:
            train_iter += 1
            writer.add_scalar('train/loss', loss.item(), train_iter)
        print('epoch {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch,
            int(batch_idx * len(youtube8m_data)),
            len(data_loader.dataset), 100. * batch_idx / len(data_loader),
            loss.item()))
    scheduler.step()

def test(args, model, data_loader, criterion, writer, ):
    global val_iter
    model.eval()
    val_loss = 0
    correct1, correct0 = 0, 0
    gt1, gt0, all_gt = 0, 0, 0
    prob_raw, gts_raw = [], []
    preds, gts = [], []
    batch_num = 1  # 0
    with torch.no_grad():
        for youtube8m_data, stft_data, label in data_loader:
            batch_num += 1
            label = label.view(-1)
            if args.use_gpu == 1:
                youtube8m_data = youtube8m_data.cuda()
                stft_data = stft_data.cuda()
                label = label.cuda()

            label = label.view(-1).cuda()
            output = model(youtube8m_data, stft_data)
            output = output.view(-1, 2)
            loss = criterion(output, label)

            if loss.item() > 0:
                writer.add_scalar('val/loss', loss.item(), val_iter)

            val_loss += loss.item()
            output = functional.softmax(output, dim=1)
            prob = output[:, 1]
            gts_raw.append(to_numpy(label))
            prob_raw.append(to_numpy(prob))

            gt = label.cpu().detach().numpy()
            prediction = np.nan_to_num(prob.squeeze().cpu().detach().numpy()) > 0.5
            idx1 = np.where(gt == 1)[0]
            idx0 = np.where(gt == 0)[0]
            gt1 += len(idx1)
            gt0 += len(idx0)
            all_gt += len(gt)
            correct1 += len(np.where(gt[idx1] == prediction[idx1])[0])
            correct0 += len(np.where(gt[idx0] == prediction[idx0])[0])
        for x in gts_raw:
            gts.extend(x.tolist())
        for x in prob_raw:
            preds.extend(x.tolist())

    val_loss /= batch_num
    ap = get_ap(gts_raw, prob_raw)
    print("AP: {:.3f}".format(ap))
    print('Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(val_loss,
                                                                   correct1 + correct0,
                                                                   all_gt,
                                                                   100. * (correct0 + correct1) / all_gt))
    print('Accuracy1: {}/{} ({:.0f}%), Accuracy0: {}/{} ({:.0f}%)'.format(
        correct1, gt1, 100. * correct1 / (gt1 + 1e-5), correct0, gt0,
                       100. * correct0 / (gt0 + 1e-5)))
    return ap.mean()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', type = int, default = 1)
    parser.add_argument('--resume', type = str, default = None)
    parser.add_argument('--samples_dir', type = str, default = '/home/tione/notebook/VideoStructuring/dataset/samples/seg')
    parser.add_argument('--model_dir', type = str, default = '/home/tione/notebook/VideoStructuring/dataset/model/seg')
    parser.add_argument('--youtube8m_cache_size', type = int, default = 100000)
    parser.add_argument('--stft_cache_size', type = int, default = 100000)
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--extract_youtube8m', type = bool, default = True)
    parser.add_argument('--extract_stft', type = bool, default = True)
    parser.add_argument('--youtube8m_ratio', type = float, default = 0.8)
    parser.add_argument('--stft_ratio', type = float, default = 0.2)
    parser.add_argument('--sim_dim', type = int, default = 512)
    parser.add_argument('--youtube8m_dim', type = int, default = 2048)
    parser.add_argument('--stft_dim', type = int, default = 512)
    parser.add_argument('--lstm_hidden_size', type = int, default = 512)
    parser.add_argument('--window_size', type = int, default = 5)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--youtube8m_feat_dim', type=int, default=2048)
    parser.add_argument('--bidirectional', type=bool, default=True)
    parser.add_argument('--stft_feat_dim', type=int, default=512)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--logs_dir', type=str, default='/home/tione/notebook/VideoStructuring/dataset/log/seg')

    args = parser.parse_args()

    seg_dataset.youtube8m_cache_size = args.youtube8m_cache_size
    seg_dataset.stft_cache_size = args.stft_cache_size

    train_samples_path = os.path.join(args.samples_dir, 'train')
    val_samples_path = os.path.join(args.samples_dir, 'test')

    train_loader = DataLoader(SegDataset(train_samples_path), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(SegDataset(val_samples_path), batch_size=args.batch_size, shuffle=False)

    model = LGSS(args)
    if args.use_gpu == 1:
        model = model.cuda()
    model = nn.DataParallel(model)
    if args.resume is not None:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])

    optimizer = Adam(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones = [10, 30])
    criterion = nn.CrossEntropyLoss(torch.Tensor([0.1, 1]))
    if args.use_gpu == 1:
        criterion = criterion.cuda()

    print("start training...")
    os.makedirs(args.model_dir)
    os.makedirs(args.logs_dir)

    writer = SummaryWriter(logdir=args.logs_dir)
    max_ap = -1
    for epoch in range(args.epochs):
        train(args, model, train_loader, optimizer, scheduler, epoch, criterion)
        print("Val Acc")
        ap = test(args, model, val_loader, criterion, mode='val')
        if ap > max_ap:
            is_best = True
            max_ap = ap
        else:
            is_best = False
        save_checkpoint(
            {
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
            }, is_best = is_best, fpath = osp.join(args.model_dir, 'checkpoint.pth.tar'))

if __name__ == '__main__':
    main()
