from utils.torch_utils import *
import torch.nn.functional as functional
import tqdm
import sklearn
import sklearn.metrics
from concurrent.futures import ThreadPoolExecutor
import json
import math

def _inference(args, model, data_youtube8m, data_stft, video_id, index, ts, label, criterion, ori_label):
    if args.use_gpu == 1:
        try:
            data_youtube8m = data_youtube8m.cuda()
            data_stft = data_stft.cuda()
            ori_label = ori_label.cuda()
        except Exception as e:
            print(e)
            return []
    output = model(data_youtube8m, data_stft)
    output = output.view(-1, 2)
    loss = criterion(output, ori_label)
    output = functional.softmax(output, dim=1)
    probs = output[:, 1].cpu().detach().numpy()
    probs = probs.tolist()
    res = []
    for i in range(len(label)):
        res.append([video_id[i], index[i], ts[i], probs[i], label[i]])
    return res, loss

def inference(args, model, data_loader, criterion):
    model.eval()
    res = []
    total_loss = 0.0
    with torch.no_grad():
        with ThreadPoolExecutor(max_workers=args.max_worker) as executor:
            for ps in gen_batch(data_loader, executor, args, model, criterion):
                for p in ps:
                    t, loss = p.result()
                    res.extend(t)
                    total_loss += loss
                    break
    return res, total_loss

def val(res, threshold, total_loss, args):
    probs = [x[3] for x in res]
    predicts = [1 if x[3] > threshold else 0 for x in res]
    labels = [x[4] for x in res]
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
        video_id = x[0]
        video_ids.add(video_id)
        index = x[1]
        ts = x[2]
        prob = x[3]
        label = x[4]
        if video_id not in video_predicts:
            video_predicts[video_id] = []
        predict = 0
        if prob > threshold:
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

def gen_batch(data_loader, executor, args, model, criterion):
    cnt = 0
    ps = []
    for data_youtube8m, data_stft, label, video_id, index, ts in tqdm.tqdm(data_loader, total = len(data_loader)):
        cnt += 1
        ps.append(executor.submit(_inference, args, model,
            data_youtube8m, data_stft,
            video_id, index.numpy().tolist(), ts.numpy().tolist(), label.numpy().tolist(), criterion, label))
        if cnt % args.max_worker == 0:
            yield ps
            ps = []
    if cnt > 0:
        yield ps
