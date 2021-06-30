from lgss.utils.torch_utils import *
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

def _inference(cfg, args, model, criterion, data_place, data_vit, data_act, data_aud, target, end_frames, video_ids, shot_ids, end_shotids):
    data_place = data_place.cuda() if 'place' in cfg.dataset.mode else []
    data_vit = data_vit.cuda() if 'vit' in cfg.dataset.mode else []
    data_act = data_act.cuda() if 'act' in cfg.dataset.mode else []
    data_aud = data_aud.cuda() if 'aud' in cfg.dataset.mode else []
    target = target.view(-1).cuda()
    
    t1 = []
    t2 = []
    t3 = []
    t4 = []

    for i in range(len(video_ids[0])):
        for j in range(len(video_ids)):
            t1.append(video_ids[j][i])
            t2.append(shot_ids[j][i])
            t3.append(end_shotids[j][i])
            t4.append(end_frames[i][j])
    video_ids = t1
    shot_ids = t2
    end_shotids = t3
    end_frames = t4
    
    labels = to_numpy(target).tolist()
    outs = model(data_place, data_vit, data_act, data_aud)
    res = {'video_ids': video_ids,
            'shot_ids': shot_ids,
            'end_shotids': end_shotids,
            'end_frames': end_frames,
            'labels': labels}
    for k, output in outs.items():
        res[k] = {}
        output = output.view(-1, 2)
        loss = criterion(output, target)
        res[k]['loss'] = loss.item()
        output = F.softmax(output, dim=1)
        prob = output[:, 1]
        probs = to_numpy(prob).tolist()
        res[k]['probs'] = probs
    return res

def inference(cfg, args, model, data_loader, criterion):
    model.eval()
    res = {}
    with torch.no_grad():
        for data_place, data_vit, data_act, data_aud, target, end_frames, video_ids, shot_ids, end_shotids in tqdm.tqdm(data_loader, total=len(data_loader), desc='inference'):
            t = _inference(cfg, args, model, criterion, data_place, data_vit, data_act, data_aud, target,
                                 end_frames, video_ids, shot_ids, end_shotids)
            for k, v in t.items():
                if k not in res:
                    res[k] = v
                else:
                    if k == 'video_ids' or k == 'shot_ids' or k == 'end_shotids' or k == 'end_frames' or k == 'labels':
                        res[k].extend(v)
                    else:
                        res[k]['loss'] += v['loss']
                        res[k]['probs'].extend(v['probs'])

    return res

def val(cfg, res, threshold, args, fps_dict):
    t = {}
    for mode in cfg.dataset.mode + ['fusion']:
        t_res = []
        for index in range(len(res['video_ids'])):
            t_res.append([
                res['labels'][index], 
                res[mode]['probs'][index], 
                res['end_frames'][index],
                res['video_ids'][index],
                res['shot_ids'][index],
                res['end_shotids'][index],
                ])
        t[mode] = _val(cfg, t_res, threshold, res[mode]['loss'], args, fps_dict)
    return t


def _val(cfg, res, threshold, total_loss, args, fps_dict):
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
    topn = args.topn
    smooth_threshold = threshold - args.smooth_threshold

    video_ids = set([])
    video_predicts = {}
    video_res = {}
    for x in res:
        video_id = x[3]
        shot_id = x[4]
        end_shotid = x[5]
        video_ids.add(video_id)
        '''
        #过滤填充id
        if shot_id > end_shotid:
            continue
        if shot_id < 0:
            continue
        '''
        #print(x[2], video_id, fps_dict[video_id])
        ts = float(x[2]) / fps_dict[video_id]
        prob = x[1]
        if video_id not in video_predicts:
            video_predicts[video_id] = []
        if video_id not in video_res:
            video_res[video_id] = []
        predict = 0
        if prob >= threshold:
            predict = 1
        if predict == 1:
            video_predicts[video_id].append(ts)
        video_res[video_id].append((ts, prob))
    video_predicts = {k: sorted(list(set(v))) for k, v in video_predicts.items()}

    t_dict = {}
    for k, v in video_predicts.items():
        if len(v) == 0 and topn > 0:
            t = sorted(video_res[k], key=lambda x: -x[1])
            l = len(t)
            for index in range(topn):
                if index < l and t[index][1] >= smooth_threshold:
                    v.append(t[index][0])
            t_dict[k] = sorted(list(set(v)))
        else:
            t_dict[k] = v

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
        #print(video_id, true_gts, predict_gts)
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
    return {'auc': auc, 'acc': acc, 'recall': recall, 'precision': precision, 'ap': ap, 'f1': f1, 'avg_loss': avg_loss, 'f1_w': f1_w}
