from utils.torch_utils import *
import torch.nn.functional as functional
import tqdm
import sklearn
import sklearn.metrics
from concurrent.futures import ThreadPoolExecutor
import json
import math
import torch.nn.functional as F
import sklearn
import sklearn.metrics

def _inference(cfg, args, model, criterion, data_place, data_cast, data_act, data_aud, target, end_frames, video_ids):
    total_loss = 0
    data_place = data_place.cuda() if 'place' in cfg.dataset.mode else []
    data_cast  = data_cast.cuda()  if 'cast'  in cfg.dataset.mode else []
    data_act   = data_act.cuda()   if 'act'   in cfg.dataset.mode else []
    data_aud   = data_aud.cuda()   if 'aud'   in cfg.dataset.mode else []
    target = target.view(-1).cuda()

    output = model(data_place, data_cast, data_act, data_aud)
    output = output.view(-1, 2)
    loss = criterion(output, target)

    total_loss += loss.item()
    output = F.softmax(output, dim=1)
    prob = output[:, 1]
    labels = to_numpy(target).tolist()
    probs = to_numpy(prob)
    end_frames = to_numpy(end_frames.view(-1)).tolist()
    video_ids = to_numpy(video_ids.view(-1)).tolist()
    other = []
    for i in range(len(labels)):
        other.append((labels[i], probs[i], end_frames[i], video_ids[i]))
    return other, total_loss

def inference(cfg, args, model, data_loader, criterion):
    model.eval()
    res = []
    total_loss = 0.0
    with torch.no_grad():
        for data_place, data_cast, data_act, data_aud, target, end_frames, video_ids in tqdm.tqdm(data_loader, total=len(data_loader)):
            t, loss = _inference(cfg, args, model, criterion, data_place, data_cast, data_act, data_aud, target, end_frames, video_ids)
            res.extend(t)
            total_loss += loss
    return res, total_loss

def val(res, threshold, total_loss, args, fps_dict):
    probs = [x[1] for x in res]
    predicts = [1 if x[1] >= threshold else 0 for x in res]
    labels = [x[0] for x in res]
    auc = sklearn.metrics.roc_auc_score(labels, probs)
    acc = sklearn.metrics.accuracy_score(labels, predicts)
    recall = sklearn.metrics.recall_score(labels, predicts, zero_division=1)
    precision = sklearn.metrics.precision_score(labels, predicts, zero_division=1)
    ap = sklearn.metrics.average_precision_score(labels, probs)
    f1 = sklearn.metrics.f1_score(labels, predicts, zero_division=1)
    avg_loss = total_loss / len(res)

    video_ids = set([])
    video_predicts = {}
    for x in res:
        video_id = x[3]
        video_ids.add(video_id)
        ts = float(x[2]) / fps_dict[video_id]
        prob = x[1]
        if video_id not in video_predicts:
            video_predicts[video_id] = []
        predict = 0
        if prob >= threshold:
            predict = 1
        if predict == 1:
            video_predicts[video_id].append(ts)
    video_predicts = {k: sorted(v) for k, v in video_predicts.items()}

    video_true = {}
    annotation_dict = {}
    with open(args.annotation_file, 'r') as f:
        annotation_dict = json.load(f)

    tp = 0
    pt = 0
    rt = 0
    for video_id in video_ids:
        true_gts = [x['segment'][1] for x in annotation_dict['{}.mp4'.format(video_id)]['annotations'][:-1]]
        predict_gts = video_predicts[video_id]
        t_id = 0
        p_id = 0
        t_len = len(true_gts)
        p_len = len(predict_gts)
        pt += p_len
        rt += t_len
        while t_id < t_len and p_id < p_len:
            t_cur = true_gts[t_id]
            p_cur = predict_gts[p_id]
            if math.fabs(p_cur - t_cur) < 0.5:
                tp += 1
                t_id += 1
                p_id += 1
            else:
                if p_cur > t_cur:
                    t_id += 1
                else:
                    p_id += 1
    t1 = 0
    if pt != 0:
        t1 = tp / pt
    t2 = tp / rt
    f1_w = 0
    if t1 + t2 > 0:
        f1_w = 2 * t1 * t2 / (t1 + t2)
    print(tp, pt, rt)
    return auc, acc, recall, precision, ap, f1, avg_loss, f1_w
