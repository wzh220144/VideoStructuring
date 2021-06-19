import lgss.data.train_data_utils as train_data_utils
import lgss.data.inference_data_utils as inference_data_util
import torch
import torch.utils.data as data
from torchvision import datasets, transforms
from utilis import read_json, read_pkl, read_txt_list, strcal
from utilis.package import *
import cv2

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
        self.fps_dict = {}
        self.shot_frames = {}
        self.frame_count_dict = {}
        video_ids = set([])
        for x in self.listIDs:
            for xx in x:
                video_ids.add(xx['imdbid'])
        shot_txt_dir = os.path.join(cfg.data_root, 'shot_txt')
        for video_id in video_ids:
            txt_path = os.path.join(shot_txt_dir, '{}.txt'.format(video_id))
            video_path = os.path.join(cfg.video_dir, '{}.mp4'.format(video_id))
            cap = cv2.VideoCapture(video_path)
            self.frame_count_dict[video_id] = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    for index, line in enumerate(f):
                        cols = line.strip('\n').split(' ')
                        self.shot_frames[self.gen_key(video_id, strcal(index, 0))] = [int(x) for x in cols]

    def gen_key(self, video_id, shot_id):
        return video_id + '_' + shot_id

    def __len__(self):
        return len(self.listIDs)

    def __getitem__(self, index):
        ID_list = self.listIDs[index]
        if isinstance(ID_list, (tuple, list)):
            place_feats, cast_feats, act_feats, aud_feats, labels, end_frames, video_ids = [], [], [], [], [], [], []
            for ID in ID_list:
                place_feat, cast_feat, act_feat, aud_feat, label, end_frame, video_id = self._get_single_item(ID)
                place_feats.append(place_feat)
                cast_feats.append(cast_feat)
                act_feats.append(act_feat)
                aud_feats.append(aud_feat)
                labels.append(label)
                end_frames.append(end_frame)
                video_ids.append(video_id)
            if 'place' in self.mode:
                place_feats = torch.stack(place_feats)
            if 'cast' in self.mode:
                cast_feats = torch.stack(cast_feats)
            if 'act' in self.mode:
                act_feats = torch.stack(act_feats)
            if 'aud' in self.mode:
                aud_feats = torch.stack(aud_feats)
            labels = np.array(labels)
            end_frames = np.array(end_frames)
            video_ids = np.array(video_ids)
            return place_feats, cast_feats, act_feats, aud_feats, labels, end_frames, video_ids
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
        end_frame = self.shot_frames[self.gen_key(imdbid, shotid)][1]
        label = 0
        if 'annos_dict' in self.data_dict:
            label = self.data_dict["annos_dict"].get(imdbid).get(shotid, 0)
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
        return place_feats, cast_feats, act_feats, aud_feats, label, end_frame, imdbid

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

def get_train_data(cfg):
    imdbidlist_json, annos_dict, annos_valid_dict, casts_dict, acts_dict = train_data_utils.data_pre(cfg)
    partition = train_data_utils.data_partition(cfg, imdbidlist_json, annos_valid_dict)
    data_dict = {"annos_dict": annos_dict, "casts_dict": casts_dict, "acts_dict": acts_dict}
    train_data = Preprocessor(cfg, partition['train'], data_dict)
    val_data = Preprocessor(cfg, partition['val'], data_dict)
    return train_data, val_data

def get_inference_data(cfg, video_names):
    data = inference_data_util.get_data(cfg, video_names)
    all_set = Preprocessor(cfg, data, {})
    return all_set
