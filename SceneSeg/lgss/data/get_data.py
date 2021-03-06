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
        self.modals = {}
        for k, v in cfg.model.modal.items():
            if k == 'fusion':
                continue
            if v.use == 1:
                self.modals[k] = v
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
        res = {'labels': [], 'end_frames': [], 'video_ids': [], 'shot_ids': [], 'end_shots': []}
        for modal in self.modals.keys():
            res['{}_feats'.format(modal)] = []
        
        for ID in ID_list:
            single_res = self._get_single_item(ID)
            for k, v in single_res.items():
                res['{}s'.format(k)].append(v)
            
        for modal in sorted(self.modals.keys()):
            res['{}_feats'.format(modal)] = torch.stack(res['{}_feats'.format(modal)])
        res['labels'] = np.array(res['labels'])
        res['end_frames']= np.array(res['end_frames'])
        t = []
        t.append('labels')
        t.append('end_frames')
        t.append('video_ids')
        t.append('shot_ids')
        t.append('end_shots')
        for modal in sorted(self.modals.keys()):
            t.append('{}_feats'.format(modal))
        t = [res[x] for x in t]
        return tuple(t)

    # ??????????????????
    def _get_single_item(self, ID):
        res = {}
        #??????vid
        imdbid = ID['imdbid']
        #??????????????????shotid
        shotid = ID['shotid']
        #?????????video???????????????shot
        end_shot = int(ID['endshot'])
        #?????????video???????????????frame
        end_frame = self.shot_frames[self.gen_key(imdbid, ID['endshot'])][1]
        if int(shotid) < 0:
            end_frame = self.shot_frames[self.gen_key(imdbid, strcal(0, 0))][1]
        elif int(shotid) <= end_shot:
            end_frame = self.shot_frames[self.gen_key(imdbid, shotid)][1]
        #?????????shot?????????label
        label = 0
        if 'annos_dict' in self.data_dict:
            label = self.data_dict["annos_dict"].get(imdbid).get(shotid, 0)
            if int(shotid) < 0:
                label = self.data_dict["annos_dict"].get(imdbid).get(strcal(0, 0), 0)
            elif int(shotid) > end_shot:
                label = self.data_dict["annos_dict"].get(imdbid).get(strcal(end_shot, 0), 0)
        #?????????shot??????feats
        for modal in self.modals.keys():
            res['{}_feat'.format(modal)] = []
            for ind in self.shot_boundary_range:
                name = self.get_feat_name(shotid, ind, end_shot)
                path = osp.join(self.data_root, '{}/{}'.format(self.modals[modal].base, imdbid), name)
                try:
                    feat = np.load(path, allow_pickle=True)
                    res['{}_feat'.format(modal)].append(torch.from_numpy(feat).float())
                except Exception as e:
                    print('{}:{}'.format(path, e))
            if len(res['{}_feat'.format(modal)]) > 0:
                res['{}_feat'.format(modal)] = torch.stack(res['{}_feat'.format(modal)])
            res['label'] = label
            res['end_frame'] = end_frame
            res['video_id'] = imdbid
            res['shot_id'] = int(shotid)
            res['end_shot'] = end_shot
        return res

    def get_feat_name(self, shotid, ind, end_shot):
        name = ''
        if int(shotid) + ind < 0:
            name = 'shot_0000.npy'
        elif int(shotid) + ind > end_shot:
            name = 'shot_{}.npy'.format(str(end_shot).zfill(4))
        else:
            name = 'shot_{}.npy'.format(strcal(shotid, ind))
        return name


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
