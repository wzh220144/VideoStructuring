from __future__ import print_function

from mmcv import Config
from tensorboardX import SummaryWriter

import lgss.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lgss.data.get_data import get_train_data
from torch.utils.data import DataLoader
from lgss.utilis import (cal_MIOU, cal_Recall, cal_Recall_time, get_ap, get_mAP_seq,
                    load_checkpoint, mkdir_ifmiss, pred2scene, save_checkpoint,
                    save_pred_seq, scene2video, to_numpy, write_json)
from utilis.package import *
import sklearn
import lgss.models.lgss_util as lgss_util
import sklearn.metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Runner')
    parser.add_argument('--config', type=str, default='/home/tione/notebook/VideoStructuring/SceneSeg/config/train_hsv.py')
    parser.add_argument('--annotation_file', type=str, default='/home/tione/notebook/dataset/structuring/GroundTruth/train5k.txt')
    args = parser.parse_args()
    return args

args = parse_args()
cfg = Config.fromfile(args.config)
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus
writer = SummaryWriter(logdir=cfg.logger.logs_dir)
fps_dict = {}

train_iter = 0

def train(cfg, model, train_loader, val_loader, optimizer, scheduler, epoch, criterion, best_f1, best_ap, best_threshold):
    global train_iter
    model.train()
    labels = []
    preds = []
    probs = []
    total_loss = 0.0
    cnt = 0
    for batch_idx, (data_place, data_cast, data_act, data_aud, target, end_frames, video_ids) in enumerate(train_loader):
        data_place = data_place.cuda() if 'place' in cfg.dataset.mode or 'image' in cfg.dataset.mode else []
        data_cast  = data_cast.cuda()  if 'cast'  in cfg.dataset.mode else []
        data_act   = data_act.cuda()   if 'act'   in cfg.dataset.mode else []
        data_aud   = data_aud.cuda()   if 'aud'   in cfg.dataset.mode else []
        target = target.view(-1).cuda()
        optimizer.zero_grad()
        output = model(data_place, data_cast, data_act, data_aud)
        output = output.view(-1, 2)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        prob = F.softmax(output, dim=1)[:, 1].detach().cpu().numpy()
        pred = np.nan_to_num(prob) > 0.5
        _label = target.cpu().numpy().tolist()
        _pred = pred.tolist()
        _prob = prob.tolist()
        preds.extend(_pred)
        probs.extend(_prob)
        labels.extend(_label)
        total_loss += loss.item()
        cnt += 1

        if cnt % 100 == 0:
            acc = sklearn.metrics.accuracy_score(labels, preds)
            recall = sklearn.metrics.recall_score(labels, preds, zero_division=1)
            precision = sklearn.metrics.precision_score(labels, preds, zero_division=1)
            auc = sklearn.metrics.roc_auc_score(labels, probs)
            ap = sklearn.metrics.average_precision_score(labels, probs)
            f1 = sklearn.metrics.f1_score(labels, preds, zero_division=1)

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
                int(batch_idx * len(data_place)),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                total_loss / cnt,
                auc,
                acc,
                recall,
                precision,
                ap,
                f1,
            ))
            labels = []
            preds = []
            probs = []
            total_loss = 0
            cnt = 0

        if batch_idx % 1000 == 0 and batch_idx != 0:
            best_f1, best_ap, best_threshold = test(cfg, model, val_loader, criterion, best_f1, best_ap, best_threshold, epoch)

    scheduler.step()
    return best_f1, best_ap, best_threshold

def test(cfg, model, val_loader, best_f1, best_ap, best_threshold, criterion, epoch):
    global fps_dict
    print('start val...')
    res, total_loss = lgss_util.inference(args, model, val_loader, criterion)

    cur_max_threshold = -1
    cur_max_f1 = 0
    is_best = False
    cur_max_ap = 0
    for threshold in np.arange(0, 1.01, 0.01).tolist():
        auc, acc, recall, precision, ap, f1, avg_loss, f1_w = lgss_util.val(res, threshold, total_loss, args, fps_dict)
        print(
            'threshold: {}, auc: {}, acc: {}, recall: {}, precision: {}, ap: {}, f1: {}, avg_loss: {}, f1_w: {}'.format(
                threshold, auc, acc, recall, precision, ap, f1, avg_loss, f1_w))
        if f1_w > best_f1:
            is_best = True
            best_f1 = f1_w
            best_threshold = threshold
        if f1_w > cur_max_f1:
            cur_max_f1 = f1_w
            cur_max_threshold = threshold
        if ap > best_ap:
            best_ap = ap
        if ap > cur_max_ap:
            cur_max_ap = ap
    print('epoch {}: \tcur_max_threshold:{:.6f}, best_threshold: {:.6f}, cur_max_f1: {:.6f}, best_f1: {:.6f}, cur_max_ap: {:.6f}, best_ap: {:.6f}'.format(
        epoch, cur_max_threshold, best_threshold,
        cur_max_f1, best_f1, cur_max_ap, best_ap))

    save_checkpoint({'state_dict': model.state_dict(), 'epoch': epoch + 1, },
                    is_best=is_best, fpath=osp.join(args.model_dir, 'checkpoint.pth.tar'))
    return best_f1, best_ap, best_threshold

def main():
    train_data, val_data = get_train_data(cfg)
    global fps_dict
    fps_dict = val_data.fps_dict
    train_loader = DataLoader(
                    train_data, batch_size=cfg.batch_size,
                    shuffle=True, **cfg.data_loader_kwargs)
    val_loader = DataLoader(
                   val_data, batch_size=cfg.batch_size,
                   shuffle=True, **cfg.data_loader_kwargs)

    model = models.__dict__[cfg.model.name](cfg).cuda()
    model = nn.DataParallel(model)

    if cfg.resume is not None:
        checkpoint = load_checkpoint(cfg.resume)
        model.load_state_dict(checkpoint['state_dict'])

    optimizer = optim.__dict__[cfg.optim.name](
        model.parameters(), **cfg.optim.setting)
    scheduler = optim.lr_scheduler.__dict__[cfg.stepper.name](
        optimizer, **cfg.stepper.setting)
    criterion = nn.CrossEntropyLoss(torch.Tensor(cfg.loss.weight).cuda())
    print("...data and model loaded")

    print("...begin training")

    best_f1 = 0
    best_ap =0
    best_threshold = 0
    for epoch in range(1, cfg.epochs + 1):
        best_f1, best_ap, best_f1_threshold = train(cfg, model, train_loader, optimizer, scheduler, epoch, criterion, best_f1, best_ap, best_threshold)
        best_f1, best_ap, best_f1_threshold = test(cfg, model, val_loader, criterion, best_f1, best_ap, best_threshold, epoch)

if __name__ == '__main__':
    main()
