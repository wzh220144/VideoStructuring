from __future__ import print_function

import sys
sys.path.append(".")

from utilis import strcal
from utilis.package import *

def data_partition(cfg, valid_shotids, video_name):
    assert(cfg.seq_len % 2 == 0)
    seq_len_half = cfg.seq_len//2
    one_mode_idxs = []
    if len(valid_shotids) <= cfg.seq_len-1:
        one_idxs = []
        for _ in range((cfg.seq_len-len(valid_shotids))//2):
            one_idxs.append({'imdbid': video_name, 'shotid': strcal(0, 0)})
        for i in range(len(valid_shotids)):
            one_idxs.append({'imdbid': video_name, 'shotid': strcal(i, 0)})
        remain = cfg.seq_len-len(one_idxs)
        for _ in range(remain):
            one_idxs.append({'imdbid': video_name, 'shotid': strcal(len(valid_shotids)-1, 0)})
        one_mode_idxs.append(one_idxs)
    else:    
        for i in range(seq_len_half-1, len(valid_shotids)-seq_len_half):
            one_idxs = []
            for idx in range(-seq_len_half+1, seq_len_half+1):
                one_idxs.append({'imdbid': video_name, 'shotid': strcal(i, idx)})
            one_mode_idxs.append(one_idxs)
    partition = {}
    partition['all'] = one_mode_idxs
    return partition

def get_shot_ids(cfg, video_name):
    data_root = cfg.data_root
    img_dir_fn = osp.join(data_root, 'shot_keyf', video_name)
    files = os.listdir(img_dir_fn)
    shotids = [int(x.split(".jpg")[0].split("_")[1]) for x in files if x.split(".jpg")[0][-1] == "1"]
    return shotids
