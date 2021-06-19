from . import read_txt_list, mkdir_ifmiss, write_json
from .package import *
import cv2
import numpy as np
from sklearn.metrics import average_precision_score
import torch
import subprocess
import collections

def get_pair_list(anno_dict):
    sort_anno_dict_key = sorted(anno_dict.keys())
    tmp = 0
    tmp_list = []
    tmp_label_list = []
    anno_list = []
    anno_label_list = []
    for key in sort_anno_dict_key:
        # print(key, value)
        value = anno_dict.get(key)
        # print(key, value)
        tmp += value
        tmp_list.append(key)
        tmp_label_list.append(value)
        if tmp == 1:
            anno_list.append(tmp_list)
            anno_label_list.append(tmp_label_list)
            tmp = 0
            tmp_list = []
            tmp_label_list = []
            continue
        # print(anno_list)
    if len(anno_list) == 0 and len(anno_dict) == 0:
        return None
    if len(anno_list) == 0:
        return [[key for key in sort_anno_dict_key]]
    if len(anno_dict) - 1 > int(anno_list[-1][-1]):
        anno_list.append([key for key in sort_anno_dict_key[int(anno_list[-1][-1]) + 1:]])
        # print(anno_list)
    while [] in anno_list:
        anno_list.remove([])
    tmp_anno_list = [anno_list[0]]
    pair_list = []
    for ind in range(len(anno_list) - 1):
        cont_count = int(anno_list[ind + 1][0]) - int(anno_list[ind][-1])
        if cont_count > 1:
            pair_list.extend(tmp_anno_list)
            tmp_anno_list = [anno_list[ind + 1]]
            continue
        tmp_anno_list.append(anno_list[ind + 1])
    pair_list.extend(tmp_anno_list)
    return pair_list

def get_demo_scene_list(cfg, anno_dict, video_name):
    pair_list = get_pair_list(anno_dict)
    scene_list = []
    for pair in pair_list:
        start_shot, end_shot = int(pair[0]), int(pair[-1])
        start_frame = shotfrmlist[start_shot].split(" ")[0]
        end_frame = shotfrmlist[end_shot].split(" ")[1]
        scene_list.append((start_frame, end_frame))

    return scene_list, pair_list

def getIntersection(interval_1, interval_2):
    assert interval_1[0] < interval_1[1], "start frame is bigger than end frame."
    assert interval_2[0] < interval_2[1], "start frame is bigger than end frame."
    start = max(interval_1[0], interval_2[0])
    end = min(interval_1[1], interval_2[1])
    if start < end:
        return (end - start)
    return 0

def getUnion(interval_1, interval_2):
    assert interval_1[0] < interval_1[1], "start frame is bigger than end frame."
    assert interval_2[0] < interval_2[1], "start frame is bigger than end frame."
    start = min(interval_1[0], interval_2[0])
    end = max(interval_1[1], interval_2[1])
    return (end - start)

def getRatio(interval_1, interval_2):
    interaction = getIntersection(interval_1, interval_2)
    if interaction == 0:
        return 0
    else:
        return interaction / getUnion(interval_1, interval_2)

def get_ap(gts_raw, preds_raw):
    gts, preds = [], []
    for gt_raw in gts_raw:
        gts.extend(gt_raw.tolist())
    for pred_raw in preds_raw:
        preds.extend(pred_raw.tolist())
    # print ("AP ",average_precision_score(gts, preds))
    return average_precision_score(np.nan_to_num(gts), np.nan_to_num(preds))

def get_mAP(loader, gts_raw, preds_raw):
    mAP, gts, preds = [], [], []
    for gt_raw in gts_raw:
        gts.extend(gt_raw.tolist())
    for pred_raw in preds_raw:
        preds.extend(pred_raw.tolist())

    n = min(len(loader.dataset), len(gts), len(preds))
    lines = []
    for i in range(n):
        one_idx = loader.dataset.listIDs[i]
        line = '{} {} {} {}'.format(one_idx['imdbid'], one_idx['shotid'], \
                                    gts[i], preds[i])
        lines.append(line)
    lines = list(sorted(lines))
    imdbs = np.array([x.split(' ')[0] for x in lines])
    # shots = np.array([x.split(' ')[1] for x in lines])
    gts = np.array([int(x.split(' ')[2]) for x in lines], np.int32)
    preds = np.array([float(x.split(' ')[3]) for x in lines], np.float32)
    movies = np.unique(imdbs)
    for movie in movies:
        index = np.where(imdbs == movie)[0]
        ap = average_precision_score(gts[index], preds[index])
        mAP.append(round(ap, 2))
    return np.mean(mAP), np.array(mAP)

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

def save_pred_seq(cfg, loader, gts, preds, logs_dir):
    mkdir_ifmiss(osp.join(logs_dir, 'pred'))
    for threshold in np.arange(0, 1.01, 0.05):
        pred_fn = osp.join(logs_dir, 'pred/pred_{:.2f}.txt'.format(threshold))
        n = min(len(loader.dataset.listIDs), len(gts) // cfg.seq_len, len(preds) // cfg.seq_len)
        tmp = np.array(preds, np.float32)
        tmp = (tmp > threshold).astype(np.int32)
        with open(pred_fn, "w") as f:
            for i in range(n):
                for j in range(cfg.seq_len):
                    one_idx = loader.dataset.listIDs[i][j]
                    f.write('{} {} {} {}\n'.format(one_idx['imdbid'], one_idx['shotid'], \
                                                   gts[i * cfg.seq_len + j], tmp[i * cfg.seq_len + j]))

def pred2scene(cfg, threshold, video_name, logs_dir):
    pred_fn = osp.join(logs_dir, 'pred/pred_{:.2f}.txt'.format(threshold))
    # print("pred_fn", pred_fn)
    pred_list = read_txt_list(pred_fn)
    # print("pred_list", pred_list)
    scene_list, pair_list = get_demo_scene_list(cfg, pred_list, video_name)
    # print("scene_list", scene_list)
    # print("pair_list", pair_list)
    scene_dict = {}
    assert len(scene_list) == len(pair_list)
    print("...pred list to scene list process")
    for scene_ind, scene_item in enumerate(scene_list):
        scene_dict.update({
            scene_ind: {
                "shot": pair_list[scene_ind],
                "frame": scene_item
            }
        })
    write_json(osp.join(logs_dir, "pred_scene.json"), scene_dict)
    print(
        '...scene dict with the information of each scene is saved in {}'.format(osp.join(logs_dir, "pred_scene.json")))
    return scene_dict, scene_list

def scene2video(cfg, scene_list, video_name):
    source_movie_fn = '{}.mp4'.format(osp.join(cfg.video_dir, video_name))
    vcap = cv2.VideoCapture(source_movie_fn)
    fps = vcap.get(cv2.CAP_PROP_FPS)
    last_frame = vcap.get(cv2.CAP_PROP_FRAME_COUNT) - 1
    out_video_dir_fn = cfg.output_root
    t = scene_list[-1][1] + 1
    if t < last_frame:
        scene_list.append([t, last_frame])
    mkdir_ifmiss(out_video_dir_fn)
    for scene_ind, scene_item in enumerate(scene_list):
        scene = str(scene_ind).zfill(4)
        start_frame = int(scene_item[0])
        end_frame = int(scene_item[1])
        start_time, end_time = start_frame / fps, end_frame / fps
        duration_time = end_time - start_time
        # out_video_fn = osp.join(out_video_dir_fn,"scene_{}.mp4".format(scene))
        out_video_fn = osp.join(out_video_dir_fn, "{}#{:02d}#{:.3f}#{:.3f}#{}.mp4".format(video_name,
                                                                                          scene_ind,
                                                                                          start_time,
                                                                                          end_time,
                                                                                          int(np.ceil(fps))))
        if osp.exists(out_video_fn):
            print(out_video_fn + ' exist.')
            continue
        else:
            print(out_video_fn + ' not exist.')
        call_list = ['ffmpeg']
        call_list += ['-loglevel', 'warning']
        call_list += ['-hide_banner']
        call_list += ['-stats']
        call_list += [
            '-y',
            '-ss',
            str(start_time),
            '-t',
            str(duration_time),
            '-i',
            source_movie_fn]
        call_list += ['-map_chapters', '-1']
        call_list += [out_video_fn]
        subprocess.call(call_list)
