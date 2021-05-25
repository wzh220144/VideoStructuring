from __future__ import print_function

import random
import sys
from multiprocessing import Manager, Pool, Process
sys.path.append(".")
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utilis import read_json, read_pkl, read_txt_list, strcal
from utilis.package import *


normalizer = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])

transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalizer])


class Preprocessor(data.Dataset):
    def __init__(self, cfg, listIDs):
        self.cfg = cfg
        self.shot_num = cfg.shot_num
        self.listIDs = listIDs
        self.padding_head = 0
        # print(listIDs)
        self.padding_tail = int(listIDs[-1][-1]["shotid"])
        # print(self.padding_tail)
        self.data_root = cfg.data_root
        self.shot_boundary_range = range(-cfg.shot_num//2+1, cfg.shot_num//2+1)
        self.mode = cfg.dataset.mode
        self.transform = transformer
        assert(len(self.mode) > 0)
    
    def __len__(self):
        return len(self.listIDs)
    
    def __getitem__(self, index):
        ID_list = self.listIDs[index]
        if isinstance(ID_list, (tuple, list)):
            place_feats, cast_feats, act_feats, aud_feats, labels = [], [], [], [], []
            for ID in ID_list:
                place_feat, cast_feat, act_feat, aud_feat, label = self._get_single_item(ID)
                place_feats.append(place_feat)
                cast_feats.append(cast_feat)
                act_feats.append(act_feat)
                aud_feats.append(aud_feat)
                labels.append(label)
            if 'place' in self.mode:
                place_feats = torch.stack(place_feats)
            if 'cast' in self.mode:
                cast_feats = torch.stack(cast_feats)
            if 'act' in self.mode:
                act_feats = torch.stack(act_feats)
            if 'aud' in self.mode:
                aud_feats = torch.stack(aud_feats)
            labels = np.array(labels)
            return place_feats, cast_feats, act_feats, aud_feats, labels
        else:
            return self._get_single_item(ID_list)
        '''
        if isinstance(ID_list, (tuple, list)):
            imgs, cast_feats, act_feats, aud_feats, labels = [], [], [], [], []
            for ID in ID_list:
                img, label = self._get_single_item(ID)
                imgs.append(img)
                labels.append(label)
            if 'image' in self.mode:
                imgs = torch.stack(imgs)
            labels = np.array(labels)
            return imgs, cast_feats, act_feats, aud_feats, labels
        else:
            return self._get_single_item(ID_list)
        '''

    def _get_single_item(self, ID):
        imdbid = ID['imdbid']
        shotid = ID['shotid']
        label = 1 # self.data_dict["annos_dict"].get(imdbid).get(shotid)
        aud_feats, place_feats = [], []
        cast_feats, act_feats = [], []
        if 'place' in self.mode:
            for ind in self.shot_boundary_range:
                if int(shotid)+ind<0:
                    name = 'shot_0000.npy'
                elif int(shotid)+ind>self.padding_tail:
                    name = 'shot_{}.npy'.format(str(self.padding_tail).zfill(4))
                else:
                    name = 'shot_{}.npy'.format(strcal(shotid, ind))
                path = osp.join(self.data_root, 'place_feat/{}'.format(imdbid), name)
                place_feat = np.load(path)
                place_feats.append(torch.from_numpy(place_feat).float())
        if 'cast' in self.mode:
            for ind in self.shot_boundary_range:
                cast_feat_raw = self.data_dict["casts_dict"].get(imdbid).get(strcal(shotid, ind))
                cast_feat = np.mean(cast_feat_raw, axis=0)
                cast_feats.append(torch.from_numpy(cast_feat).float())
        if 'act' in self.mode:
            for ind in self.shot_boundary_range:
                act_feat = self.data_dict["acts_dict"].get(imdbid).get(strcal(shotid, ind))
                act_feats.append(torch.from_numpy(act_feat).float())
        if 'aud' in self.mode:
            for ind in self.shot_boundary_range:
                if int(shotid)+ind<0:
                    name = 'shot_0000.npy'
                elif int(shotid)+ind>self.padding_tail:
                    name = 'shot_{}.npy'.format(str(self.padding_tail).zfill(4))
                else:
                    name = 'shot_{}.npy'.format(strcal(shotid,ind))
                path = osp.join(
                    self.data_root, 'aud_feat/{}'.format(imdbid), name)
                aud_feat = np.load(path)
                aud_feats.append(torch.from_numpy(aud_feat).float())

        if len(place_feats) > 0:
            place_feats = torch.stack(place_feats)
        if len(cast_feats) > 0:
            cast_feats = torch.stack(cast_feats)
        if len(act_feats) > 0:
            act_feats = torch.stack(act_feats)
        if len(aud_feats) > 0:
            aud_feats = torch.stack(aud_feats)
        return place_feats, cast_feats, act_feats, aud_feats, label
        
        '''
        shotid = ID["shotid"]
        imgs = []
        label = 1  # this is a pesudo label
        if 'image' in self.mode:
            for ind in self.shot_boundary_range:
                if int(shotid) + ind < self.padding_head:
                    name = 'shot_{}_img_1.jpg'.format(strcal(self.padding_head, 0))
                elif int(shotid) + ind > self.padding_tail:
                    name = 'shot_{}_img_1.jpg'.format(strcal(self.padding_tail, 0))
                else:
                    name = 'shot_{}_img_1.jpg'.format(strcal(shotid, ind))
                path = osp.join(
                    self.cfg.data_root, 'shot_keyf', self.cfg.video_name, name)
                img = Image.open(path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                imgs.append(img)

        if len(imgs) > 0:
            imgs = torch.stack(imgs)
        return imgs, label
        '''

def data_partition(cfg, valid_shotids, video_name):
    assert(cfg.seq_len % 2 == 0)
    seq_len_half = cfg.seq_len//2
    one_mode_idxs = []
    idxs = []
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
    '''    
    shotid_tmp = 0
    for shotid in valid_shotids:
        if int(shotid) < shotid_tmp+seq_len_half:
            continue
        shotid_tmp = int(shotid)+seq_len_half
        one_idxs = []
        for idx in range(-seq_len_half+1, seq_len_half+1):
            one_idxs.append({'imdbid': cfg.video_name, 'shotid': strcal(shotid, idx)})
            # print(shotid, idx, one_idxs)
        one_mode_idxs.append(one_idxs)
    '''
    idxs.append(one_mode_idxs)
    #print(idxs)
    partition = {}
    partition['train'] = idxs[0]
    partition['test'] = idxs[0]
    partition['val'] = idxs[0]
    return partition


def data_pre(cfg, video_name):
    data_root = cfg.data_root
    img_dir_fn = osp.join(data_root, 'shot_keyf', video_name)
    print(img_dir_fn)
    # print(img_dir_fn)
    win_len = cfg.seq_len + cfg.shot_num  # - 1

    files = os.listdir(img_dir_fn)
    shotids = [int(x.split(".jpg")[0].split("_")[1]) for x in files if x.split(".jpg")[0][-1] == "1"]
    # print(shotids)
    '''
    to_be_del = []
    for shotid in shotids:
        del_flag = False
        for idx in range(-(win_len)//2+1, win_len//2+1):
            if ((shotid + idx) not in shotids):
                del_flag = True
                break
        if del_flag:
            to_be_del.append(shotid)
        # print(shotid, to_be_del)
    valid_shotids = []
    for shotid in shotids:
        if shotid in to_be_del:
            continue
        else:
            valid_shotids.append(shotid)
    '''
    # print(shotids)
    # return valid_shotids
    return shotids

def main():
    from mmcv import Config
    cfg = Config.fromfile("./config/demo.py")

    valid_shotids = data_pre(cfg)
    partition = data_partition(cfg, valid_shotids)
    batch_size = cfg.batch_size
    testSet = Preprocessor(cfg, partition["test"])
    test_loader = DataLoader(
                testSet, batch_size=batch_size,
                shuffle=False, **cfg.data_loader_kwargs)

    dataloader = test_loader
    for batch_idx, (data_place, data_cast, data_act, data_aud, target) in enumerate(dataloader):
        print(data_place.shape)  # bs, seq_len, shot_num, 3, 224, 224
        print(batch_idx, target.shape)
        # if batch_idx > 1:
        #     break
        # pdb.set_trace()
    pdb.set_trace()


if __name__ == '__main__':
    main()
