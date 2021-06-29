import lgss.data.train_data_utils as train_data_utils
import lgss.data.inference_data_utils as inference_data_util
import torch
import torch.utils.data as data
from torchvision import datasets, transforms
from lgss.utils import read_json, read_pkl, read_txt_list, strcal
from lgss.utils.package import *
import cv2
import tqdm

class Preprocessor(data.Dataset):
    def __init__(self, cfg, listIDs, data_dict):
        self.cfg = cfg
        self.shot_num = cfg.shot_num
        self.listIDs = listIDs
        self.padding_head = 0
        # print(listIDs)
        self.data_root = cfg.data_root
        self.shot_boundary_range = range(-cfg.shot_num // 2 + 1, cfg.shot_num // 2 + 1)
        self.mode = cfg.dataset.mode
        assert (len(self.mode) > 0)
        self.data_dict = data_dict
        self.fps_dict = {}
        self.shot_frames = {}
        self.frame_count_dict = {}
        video_ids = set([])
        for x in self.listIDs:
            for xx in x:
                video_ids.add(xx['imdbid'])
        shot_txt_dir = os.path.join(cfg.data_root, 'shot_txt')
        video_info_dict = {}
        with open(cfg.video_dir + '.info', 'r') as f:
            video_info_dict = json.load(f)
        for video_id in tqdm.tqdm(video_ids):
            txt_path = os.path.join(shot_txt_dir, '{}.txt'.format(video_id))
            video_path = os.path.join(cfg.video_dir, '{}.mp4'.format(video_id))
            self.frame_count_dict[video_id] = video_info_dict[video_id]['frame_count']
            self.fps_dict[video_id] = video_info_dict[video_id]['fps']
            if os.path.exists(txt_path):
                #print(txt_path)
                with open(txt_path, 'r') as f:
                    for index, line in enumerate(f):
                        #print(index, line)
                        cols = line.strip('\n').split(' ')
                        self.shot_frames[self.gen_key(video_id, strcal(index, 0))] = [int(x) for x in cols]

    def gen_key(self, video_id, shot_id):
        return video_id + '_' + shot_id

    def __len__(self):
        return len(self.listIDs)

    def __getitem__(self, index):
        ID_list = self.listIDs[index]
        if isinstance(ID_list, (tuple, list)):
            place_feats, vit_feats, act_feats, aud_feats, labels, end_frames, video_ids, shot_ids, end_shotids = [], [], [], [], [], [], [], [], []
            for ID in ID_list:
                place_feat, vit_feat, act_feat, aud_feat, label, end_frame, video_id, shot_id, end_shotid = self._get_single_item(ID)
                place_feats.append(place_feat)
                vit_feats.append(vit_feat)
                act_feats.append(act_feat)
                aud_feats.append(aud_feat)
                labels.append(label)
                end_frames.append(end_frame)
                video_ids.append(video_id)
                shot_ids.append(shot_id)
                end_shotids.append(end_shotid)
            if 'place' in self.mode:
                place_feats = torch.stack(place_feats)
            if 'vit' in self.mode:
                vit_feats = torch.stack(vit_feats)
            if 'act' in self.mode:
                act_feats = torch.stack(act_feats)
            if 'aud' in self.mode:
                aud_feats = torch.stack(aud_feats)
            labels = np.array(labels)
            end_frames = np.array(end_frames)
            video_ids = video_ids
            return place_feats, vit_feats, act_feats, aud_feats, labels, end_frames, video_ids, shot_ids, end_shotids
        else:
            return self._get_single_item(ID_list)

    # 得到单条样本
    def _get_single_item(self, ID):
        #得到vid
        imdbid = ID['imdbid']
        #得到当前主要shotid
        shotid = ID['shotid']
        #得到该video的最后一个shot
        end_shot = int(ID['endshot'])
        #得到该video的最后一个frame
        end_frame = self.shot_frames[self.gen_key(imdbid, ID['endshot'])][1]
        if int(shotid) < 0:
            end_frame = self.shot_frames[self.gen_key(imdbid, strcal(0, 0))][1]
        elif int(shotid) <= end_shot:
            end_frame = self.shot_frames[self.gen_key(imdbid, shotid)][1]
        #得到该shot对应的label
        label = 0
        if 'annos_dict' in self.data_dict:
            label = self.data_dict["annos_dict"].get(imdbid).get(shotid, 0)
            if int(shotid) < 0:
                label = self.data_dict["annos_dict"].get(imdbid).get(strcal(0, 0), 0)
            elif int(shotid) > end_shot:
                label = self.data_dict["annos_dict"].get(imdbid).get(strcal(end_shot, 0), 0)
        #得到该shot对应feats
        aud_feats = []
        place_feats = []
        vit_feats = []
        act_feats = []
        if 'place' in self.mode:
            for ind in self.shot_boundary_range:
                name = self.get_feat_name(shotid, ind, end_shot)
                path = osp.join(self.data_root, '{}/{}'.format(self.cfg.place_base, imdbid), name)
                place_feat = np.load(path, allow_pickle=True)
                place_feats.append(torch.from_numpy(place_feat).float())
        if 'vit' in self.mode:
            for ind in self.shot_boundary_range:
                name = self.get_feat_name(shotid, ind, end_shot)
                path = osp.join(self.data_root, '{}/{}'.format(self.cfg.vit_base, imdbid), name)
                vit_feat = np.load(path, allow_pickle=True)
                vit_feats.append(torch.from_numpy(vit_feat).float())
        if 'aud' in self.mode:
            for ind in self.shot_boundary_range:
                name = self.get_feat_name(shotid, ind, end_shot)
                try:
                    path = osp.join(self.data_root, '{}/{}'.format(self.cfg.aud_baseimdbid), name)
                    aud_feat = np.load(path)
                    aud_feats.append(torch.from_numpy(aud_feat).float())
                except Exception as e:
                    print('{}:{}'.format(path, e))

        if len(place_feats) > 0:
            place_feats = torch.stack(place_feats)
        if len(vit_feats) > 0:
            vit_feats = torch.stack(vit_feats)
        if len(act_feats) > 0:
            act_feats = torch.stack(act_feats)
        if len(aud_feats) > 0:
            aud_feats = torch.stack(aud_feats)
        return place_feats, vit_feats, act_feats, aud_feats, label, end_frame, imdbid, int(shotid), end_shot

    def get_feat_name(self, shotid, ind, end_shot):
        if int(shotid) + ind < 0:
            name = 'shot_0000.npy'
        elif int(shotid) + ind > end_shot:
            name = 'shot_{}.npy'.format(str(end_shot).zfill(4))
        else:
            name = 'shot_{}.npy'.format(strcal(shotid, ind))


def get_train_data(cfg):
    print('start data pre')
    imdbidlist_json, annos_dict, annos_valid_dict, vits_dict, acts_dict = train_data_utils.data_pre(cfg)
    print('start data partition')
    partition = train_data_utils.data_partition(cfg, imdbidlist_json, annos_valid_dict)
    data_dict = {"annos_dict": annos_dict, "vits_dict": vits_dict, "acts_dict": acts_dict}
    print('start train data pre process')
    train_data = Preprocessor(cfg, partition['train'], data_dict)
    print('start val data pre process')
    val_data = Preprocessor(cfg, partition['val'], data_dict)
    return train_data, val_data

def get_inference_data(cfg, video_names):
    data = inference_data_util.get_data(cfg, video_names)
    all_set = Preprocessor(cfg, data, {})
    return all_set
