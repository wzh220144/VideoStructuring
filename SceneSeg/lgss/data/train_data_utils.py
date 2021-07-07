from __future__ import print_function
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Manager, Process
from lgss.utils import read_json, read_txt_list, strcal
from lgss.utils.package import *
import tqdm

def data_partition(cfg, imdbidlist_json, annos_dict):
    assert (cfg.seq_len % 2 == 0)
    seq_len_half = cfg.seq_len // 2

    idxs = []
    for mode in ['train', 'val']:
        one_mode_idxs = []
        for imdbid in imdbidlist_json[mode]:
            anno_dict = annos_dict[imdbid]
            shotid_list = sorted(anno_dict.keys())
            end_shot = shotid_list[-1]
            if len(shotid_list) % cfg.seq_len:
                new_len = (len(shotid_list) // cfg.seq_len + 1) * cfg.seq_len
                padding_left = (new_len - len(shotid_list)) // 2
                padding_right = new_len - len(shotid_list) - padding_left
                shotid_list = [strcal(-1, 0)] * padding_left + shotid_list + [strcal(int(shotid_list[-1]) + 1, 0)] * padding_right
            for i in range(len(shotid_list) // cfg.seq_len):
                one_idxs = []
                for j in range(cfg.seq_len):
                    #one_idxs.append({'imdbid': imdbid, 'shotid': strcal(cfg.seq_len * i, j), "endshot": end_shot})
                    one_idxs.append({'imdbid': imdbid, 'shotid': shotid_list[cfg.seq_len * i + j], "endshot": end_shot})
                one_mode_idxs.append(one_idxs)
        idxs.append(one_mode_idxs)
    partition = {}
    partition['train'] = idxs[0]
    partition['val'] = idxs[1]
    return partition

def data_pre_one(cfg, imdbid, acts_dict, casts_dict, annos_dict, annos_valid_dict):
    data_root = cfg.data_root
    label_fn = osp.join(data_root, 'labels')
    place_feat_fn = osp.join(data_root, 'place_feat')
    win_len = cfg.seq_len + cfg.shot_num  # - 1

    files = os.listdir(osp.join(place_feat_fn, imdbid))
    all_shot_place_feat = [int(x.split('.')[0].split('_')[1]) for x in files]

    anno_fn = '{}/{}.txt'.format(label_fn, imdbid)
    anno_dict = get_anno_dict(anno_fn)
    # print(imdbid, anno_dict)
    annos_dict.update({imdbid: anno_dict})
    # get anno_valid_dict
    anno_valid_dict = anno_dict.copy()

    shotids = [int(x) for x in anno_valid_dict.keys()]
    to_be_del = []
    '''
    for shotid in shotids:
        del_flag = False
        for idx in range(-(win_len)//2+1, win_len//2+1):
            if ((shotid + idx) not in all_shot_place_feat) or \
                 anno_dict.get(str(shotid+idx).zfill(4)) is None:
                del_flag = True
                break
        if del_flag:
            to_be_del.append(shotid)
    '''
    for shotid in to_be_del:
        del anno_valid_dict[str(shotid).zfill(4)]
    annos_valid_dict.update({imdbid: anno_valid_dict})
    #########################
    # act_feat_fn = osp.join(data_root, "act_feat/{}.pkl".format(imdbid))
    # acts_dict.update({imdbid: read_pkl(act_feat_fn)})

    # cast_feat_fn = osp.join(data_root, "cast_feat/{}.pkl".format(imdbid))
    # casts_dict.update({imdbid: read_pkl(cast_feat_fn)})


def data_pre(cfg):
    data_root = cfg.data_root
    imdbidlist_json = osp.join(data_root, 'meta/split.json')

    imdbidlist_json = read_json(imdbidlist_json)
    # print(imdbidlist_json)
    imdbidlist = imdbidlist_json['all']
    # print(imdbidlist)
    mgr = Manager()
    acts_dict_raw = mgr.dict()
    casts_dict_raw = mgr.dict()
    annos_dict_raw = mgr.dict()
    annos_valid_dict_raw = mgr.dict()
    ps = []
    with ThreadPoolExecutor(max_workers = 30) as executor:
        for imdbid in imdbidlist:
            ps.append(executor.submit(data_pre_one, cfg, imdbid, acts_dict_raw, casts_dict_raw, annos_dict_raw, annos_valid_dict_raw))
        for p in tqdm.tqdm(ps):
            p.result()

    annos_dict, annos_valid_dict = {}, {}
    acts_dict, casts_dict = {}, {}
    for key, value in annos_dict_raw.items():
        annos_dict.update({key: value})
    for key, value in annos_valid_dict_raw.items():
        annos_valid_dict.update({key: value})
    for key, value in acts_dict_raw.items():
        acts_dict.update({key: value})
    for key, value in casts_dict_raw.items():
        casts_dict.update({key: value})

    return imdbidlist_json, annos_dict, annos_valid_dict, casts_dict, acts_dict


def get_anno_dict(anno_fn):
    contents = read_txt_list(anno_fn)
    anno_dict = {}
    for content in contents:
        shotid = content.split(' ')[0]
        value = int(content.split(' ')[1])
        if value >= 0:
            anno_dict.update({shotid: value})
        elif value == -1:
            anno_dict.update({shotid: 1})
    return anno_dict
