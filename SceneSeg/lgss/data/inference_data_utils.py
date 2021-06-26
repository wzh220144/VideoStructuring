from __future__ import print_function
from lgss.utils import strcal
from lgss.utils.package import *

def get_data(cfg, video_names):
    assert (cfg.seq_len % 2 == 0)
    seq_len_half = cfg.seq_len // 2
    data_root = cfg.data_root
    one_mode_idxs = []
    for video_name in video_names:
        shotid_list = []
        with open(osp.join(data_root, 'shot_txt', video_name + '.txt'), 'r') as f:
            cur_shot_id = 0
            for line in f:
                shotid_list.append(cur_shot_id)
        if len(shotid_list) <= 1:
            continue
        shotid_list = shotid_list[:-1]
        end_shot = shotid_list[-1]

        if len(shotid_list) % cfg.seq_len > 0:
            new_len = (len(shotid_list) // cfg.seq_len + 1) * cfg.seq_len
            padding_left = (new_len - len(shotid_list)) // 2
            padding_right = new_len - len(shotid_list) - padding_left
            shotid_list = [shotid_list[0]] * padding_left + shotid_list + [shotid_list[-1]] * padding_right
        for i in range(len(shotid_list) // cfg.seq_len):
                one_idxs = []
                for j in range(cfg.seq_len):
                    one_idxs.append({'imdbid': video_name, 'shotid': strcal(cfg.seq_len * i, j), "endshot":  strcal(end_shot, 0)})
                one_mode_idxs.append(one_idxs)
    return one_mode_idxs
