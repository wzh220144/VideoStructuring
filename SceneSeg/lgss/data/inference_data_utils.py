from __future__ import print_function
from lgss.utils import strcal
from lgss.utils.package import *

def get_data(cfg, video_names):
    assert (cfg.seq_len % 2 == 0)
    seq_len_half = cfg.seq_len // 2
    data_root = cfg.data_root
    res = []
    for video_name in video_names:
        shot_ids = []
        with open(osp.join(data_root, 'shot_txt', video_name + '.txt'), 'r') as f:
            cur_shot_id = 0
            for line in f:
                shot_ids.append(cur_shot_id)
        end_shot = shot_ids[-1]
        if len(shot_ids) <= cfg.seq_len - 1:
            one_idxs = []
            for _ in range((cfg.seq_len - len(shot_ids)) // 2):
                one_idxs.append({'imdbid': video_name, 'shotid': strcal(0, 0), 'endshot': end_shot})
            for i in range(len(shot_ids)):
                one_idxs.append({'imdbid': video_name, 'shotid': strcal(i, 0), 'endshot': end_shot})
            remain = cfg.seq_len - len(one_idxs)
            for _ in range(remain):
                one_idxs.append({'imdbid': video_name, 'shotid': strcal(len(shot_ids) - 1, 0), 'endshot': end_shot})
            res.append(one_idxs)
        else:
            for i in range(seq_len_half - 1, len(shot_ids) - seq_len_half):
                one_idxs = []
                for idx in range(-seq_len_half + 1, seq_len_half + 1):
                    one_idxs.append({'imdbid': video_name, 'shotid': strcal(i, idx), 'endshot': end_shot})
                res.append(one_idxs)
    return res
