from __future__ import print_function

from mmcv import Config
from tensorboardX import SummaryWriter

import lgss.models.lgss as lgss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lgss.data.get_data import get_train_data
from torch.utils.data import DataLoader
from lgss.utils import (cal_MIOU, cal_Recall, cal_Recall_time, get_ap, get_mAP_seq, load_checkpoint, mkdir_ifmiss,
                        save_checkpoint)
from lgss.utils.package import *
import sklearn
import lgss.models.lgss_util as lgss_util
import sklearn.metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Runner')
    parser.add_argument('--config', type=str,
                        default='/home/tione/notebook/VideoStructuring/SceneSeg/lgss/config/train_hsv.py')
    parser.add_argument('--annotation_file', type=str, default='/home/tione/notebook/dataset/GroundTruth/train5k.txt')
    parser.add_argument('--topn', type=int, default=-1)
    parser.add_argument('--use_gpu', type=int, default=1)
    parser.add_argument('--smooth_threshold', type=float, default=0.1)
    args = parser.parse_args()
    return args

args = parse_args()
cfg = Config.fromfile(args.config)
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus
writer = SummaryWriter(logdir=cfg.logger.logs_dir)
fps_dict = {}

train_iter = 0

def train(cfg, model, train_loader, val_loader, optimizer, scheduler, epoch, best_f1, best_ap, best_threshold):
    loss_weight = torch.Tensor(cfg.loss.weight)
    if cfg.use_gpu == 1:
        loss_weight = loss_weight.cuda().to('cuda:0')
    criterion = nn.CrossEntropyLoss(loss_weight)

    global train_iter
    model.train()
    train_middle_data = {}
    modals = []
    for modal, config in cfg.model.modal.items():
        if modal == 'fusion':
            continue
        if config.use == 1:
            modals.append(modal)
    modals = sorted(modals)
    cnt = 0
    train_middle_data['labels'] = []
    for modal in modals + ['fusion']:
        train_middle_data['{}_preds'.format(modal)] = []
        train_middle_data['{}_probs'.format(modal)] = []
        train_middle_data['{}_total_loss'.format(modal)] = 0

    for batch_idx, datas in enumerate(train_loader):
        feat_dict = {}
        labels = datas[0]
        feat_dict['labels'] = labels.view(-1).cuda().to('cuda:0')
        end_frames = datas[1]
        feat_dict['end_frames'] = end_frames
        video_ids = datas[2]
        feat_dict['video_ids'] = video_ids
        shot_ids = datas[3]
        feat_dict['shot_ids'] = shot_ids
        end_shots = datas[4]
        feat_dict['end_shots'] = end_shots

        for index in range(len(modals)):
            feat_dict['{}_feats'.format(modals[index])] = datas[5 + index].cuda().to('cuda:0')

        outs = model(feat_dict)

        for modal in modals + ['fusion']:
            outs['{}_logits'.format(modal)] = outs['{}_logits'.format(modal)].view(-1, 2)
            outs['{}_prob'.format(modal)] = F.softmax(outs['{}_logits'.format(modal)], dim=1)[:, 1].detach().cpu().numpy()
            outs['{}_pred'.format(modal)] = np.nan_to_num(outs['{}_prob'.format(modal)]) > cfg.threshold

        optimizer.zero_grad()
        total_loss = 0
        for modal in modals + ['fusion']:
            outs['{}_loss'.format(modal)] = criterion(outs['{}_logits'.format(modal)], feat_dict['labels'])
            total_loss = total_loss + outs['{}_loss'.format(modal)] * cfg.model.modal[modal].loss_weight
        total_loss.backward()
        optimizer.step()
        
        train_middle_data['labels'].extend(feat_dict['labels'].cpu().numpy().tolist())
        for modal in modals + ['fusion']:
            train_middle_data['{}_preds'.format(modal)].extend(outs['{}_pred'.format(modal)].tolist())
            train_middle_data['{}_probs'.format(modal)].extend(outs['{}_prob'.format(modal)].tolist())
            train_middle_data['{}_total_loss'.format(modal)] += outs['{}_loss'.format(modal)].item()
        cnt += 1

        if cnt % 100 == 0:
            train_metric = {}
            labels = train_middle_data['labels']
            for modal in modals + ['fusion']:
                preds = train_middle_data['{}_preds'.format(modal)]
                probs = train_middle_data['{}_probs'.format(modal)]
                total_loss = train_middle_data['{}_total_loss'.format(modal)]

                acc = sklearn.metrics.accuracy_score(labels, preds)
                recall = sklearn.metrics.recall_score(labels, preds, zero_division=1)
                precision = sklearn.metrics.precision_score(labels, preds, zero_division=1)
                auc = sklearn.metrics.roc_auc_score(labels, probs)
                ap = sklearn.metrics.average_precision_score(labels, probs)
                f1 = sklearn.metrics.f1_score(labels, preds, zero_division=1)

                writer.add_scalar('train/{}/loss'.format(modal), total_loss / cnt, train_iter)
                writer.add_scalar('train/{}/auc'.format(modal), auc, train_iter)
                writer.add_scalar('train/{}/acc'.format(modal), acc, train_iter)
                writer.add_scalar('train/{}/recall'.format(modal), recall, train_iter)
                writer.add_scalar('train/{}/precision'.format(modal), precision, train_iter)
                writer.add_scalar('train/{}/ap'.format(modal), ap, train_iter)
                writer.add_scalar('train/{}/f1'.format(modal), f1, train_iter)

                t = 'epoch {} for {}: [{:.0f}%]\t' + \
                        'loss: {:.6f}, auc: {:.6f}, acc: {:.6f}, recall: {:.6f}, precision: {:.6f}, ap: {:.6f}, f1: {:.6f}'
                print(t.format(
                    epoch, modal,
                    100. * batch_idx / len(train_loader),
                    total_loss / cnt,
                    auc,
                    acc,
                    recall,
                    precision,
                    ap,
                    f1,
                ))
            train_middle_data['labels'] = []
            for modal in modals + ['fusion']:
                train_middle_data['{}_preds'.format(modal)] = []
                train_middle_data['{}_probs'.format(modal)] = []
                train_middle_data['{}_total_loss'.format(modal)] = 0
            cnt = 0

    scheduler.step()
    return best_f1, best_ap, best_threshold

def test(cfg, model, val_loader, best_f1, best_ap, best_threshold, epoch):
    global fps_dict
    print('start val...')
    res = lgss_util.inference(cfg, args, model, val_loader)

    cur_max_threshold = -1
    cur_max_f1 = 0
    is_best = False
    cur_max_ap = 0
    for threshold in np.arange(0, 1.01, 0.01).tolist():
        t = lgss_util.val(cfg, res, threshold, args, fps_dict)
        print('{}: {}'.format(threshold, json.dumps(t)))
        f1_w = t['fusion']['f1_w']
        ap = t['fusion']['ap']
        if f1_w > best_f1:
            is_best = True
            best_f1 = f1_w
            best_threshold = threshold
        if f1_w > cur_max_f1:
            cur_max_f1 = f1_w
            cur_max_threshold = threshold
        if ap > best_ap:
            is_best = True
            best_ap = ap
        if ap > cur_max_ap:
            cur_max_ap = ap
    print(
        'epoch {}: \tcur_max_threshold:{:.6f}, best_threshold: {:.6f}, cur_max_f1: {:.6f}, best_f1: {:.6f}, cur_max_ap: {:.6f}, best_ap: {:.6f}'.format(
            epoch, cur_max_threshold, best_threshold,
            cur_max_f1, best_f1, cur_max_ap, best_ap))

    state_dict = model.module.state_dict()
    save_checkpoint({'state_dict': state_dict, 'epoch': epoch, 'f1': cur_max_f1, 'threshold': cur_max_threshold, 'ap': ap}, is_best, epoch + 1, cfg.model_path)
    return best_f1, best_ap, best_threshold

def main():
    train_data, val_data = get_train_data(cfg)
    global fps_dict
    fps_dict = val_data.fps_dict
    train_loader = DataLoader(
        train_data, batch_size=cfg.batch_size,
        shuffle=True, **cfg.data_loader_kwargs)
    val_loader = DataLoader(
        val_data, batch_size=int(cfg.batch_size / 4),
        shuffle=True, **cfg.data_loader_kwargs)

    model = lgss.LGSS(cfg).cuda()
    model = nn.DataParallel(model)
    model = model.to('cuda:0')
    '''
    paras = list(model.parameters())
    for num, para in enumerate(paras):
        print('number:',num)
        print(para)
    '''

    if cfg.resume is not None:
        checkpoint = load_checkpoint(cfg.resume)
        model.load_state_dict(checkpoint['state_dict'])

    optimizer = optim.__dict__[cfg.optim.name](
        model.parameters(), **cfg.optim.setting)

    scheduler = optim.lr_scheduler.__dict__[cfg.stepper.name](
        optimizer, **cfg.stepper.setting)
    print("...data and model loaded")

    print("...begin training")

    best_f1 = 0
    best_ap = 0
    best_threshold = 0
    for epoch in range(1, cfg.epochs + 1):
        best_f1, best_ap, best_threshold = train(cfg, model, train_loader, val_loader, optimizer, scheduler, epoch, best_f1, best_ap, best_threshold)
        best_f1, best_ap, best_threshold = test(cfg, model, val_loader, best_f1, best_ap, best_threshold, epoch)

if __name__ == '__main__':
    main()

