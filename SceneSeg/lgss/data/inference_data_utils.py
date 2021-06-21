from __future__ import print_function
from lgss.utils import strcal
from lgss.utils.package import *

def get_data(cfg, video_names):
    assert (cfg.seq_len % 2 == 0)
    seq_len_half = cfg.seq_len // 2
    data_root = cfg.data_root
    res = []
    for video_name in video_names:
        img_dir_fn = osp.join(data_root, 'shot_keyf', video_name)
        files = os.listdir(img_dir_fn)
        shot_ids = [int(x.split(".jpg")[0].split("_")[1]) for x in files if x.split(".jpg")[0][-1] == "1"]
        if len(shot_ids) <= cfg.seq_len - 1:
            one_idxs = []
            for _ in range((cfg.seq_len - len(shot_ids)) // 2):
                one_idxs.append({'imdbid': video_name, 'shotid': strcal(0, 0)})
            for i in range(len(shot_ids)):
                one_idxs.append({'imdbid': video_name, 'shotid': strcal(i, 0)})
            remain = cfg.seq_len - len(one_idxs)
            for _ in range(remain):
                one_idxs.append({'imdbid': video_name, 'shotid': strcal(len(shot_ids) - 1, 0)})
            res.append(one_idxs)
        else:
            for i in range(seq_len_half - 1, len(shot_ids) - seq_len_half):
                one_idxs = []
                for idx in range(-seq_len_half + 1, seq_len_half + 1):
                    one_idxs.append({'imdbid': video_name, 'shotid': strcal(i, idx)})
                res.append(one_idxs)
    return res
