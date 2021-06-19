import lgss.data.train_data_utils as train_data_utils
import lgss.data.inference_data_utils as inference_data_util
import torch
import torch.utils.data as data
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
    def __init__(self, cfg, listIDs, data_dict):
        self.cfg = cfg
        self.shot_num = cfg.shot_num
        self.listIDs = listIDs
        self.padding_head = 0
        # print(listIDs)
        self.padding_tail = int(listIDs[-1][-1]["shotid"])
        # print(self.padding_tail)
        self.data_root = cfg.data_root
        self.shot_boundary_range = range(-cfg.shot_num // 2 + 1, cfg.shot_num // 2 + 1)
        self.mode = cfg.dataset.mode
        self.transform = transformer
        assert (len(self.mode) > 0)
        self.data_dict = data_dict

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
        label = 1  # self.data_dict["annos_dict"].get(imdbid).get(shotid)
        aud_feats, place_feats = [], []
        cast_feats, act_feats = [], []
        if 'place' in self.mode:
            for ind in self.shot_boundary_range:
                if int(shotid) + ind < 0:
                    name = 'shot_0000.npy'
                elif int(shotid) + ind > self.padding_tail:
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
                if int(shotid) + ind < 0:
                    name = 'shot_0000.npy'
                elif int(shotid) + ind > self.padding_tail:
                    name = 'shot_{}.npy'.format(str(self.padding_tail).zfill(4))
                else:
                    name = 'shot_{}.npy'.format(strcal(shotid, ind))
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

def get_train_data(cfg, video_name):
    imdbidlist_json, annos_dict, annos_valid_dict, casts_dict, acts_dict = train_data_utils.data_pre(cfg)
    partition = train_data_utils.data_partition(cfg, imdbidlist_json, annos_valid_dict)
    data_dict = {"annos_dict": annos_dict, "casts_dict": casts_dict, "acts_dict": acts_dict}
    train_set = Preprocessor(cfg, partition['train'], data_dict)
    val_set = Preprocessor(cfg, partition['val'], data_dict)
    return train_set, val_set

def get_inference_data(cfg, video_name):
    valid_shotids = inference_data_util.get_shot_ids(cfg, video_name)
    partition = inference_data_util.data_partition(cfg, valid_shotids, video_name)
    all_set = Preprocessor(cfg, partition['all'], {})
    return all_set
