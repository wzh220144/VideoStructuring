import torch
import os
import json
from torch.utils.data import DataLoader, Dataset
from cachetools import cached, LRUCache, TTLCache
import numpy as np
import torch
import glob
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import tqdm

youtube8m_cache_size = 100000
stft_cache_size = 100000

class SegDataset(Dataset):
    def __init__(self, path, extract_youtube8m, extract_stft, postfix, feats_dir, use_cache):
        self.samples = []
        with open(path, 'r') as fs:
            for line in fs:
                line = line.strip('\n')
                sample = json.loads(line)
                self.samples.append(sample)
        self.extract_youtube8m = extract_youtube8m
        self.extract_stft = extract_stft
        self.postfix = postfix
        self.feats_dir = feats_dir
        self.youtube8m_feats_dir = os.path.join(self.feats_dir, postfix, 'youtube8m')
        self.stft_feats_dir = os.path.join(self.feats_dir, postfix, 'stft')
        '''
        if use_cache:
            self.pre()
        '''
        self.use_cache = use_cache

    def pre(self):
        print('pre start')
        with ThreadPoolExecutor(max_workers=50) as executor:
            ps = []
            for feat_path in os.listdir(self.youtube8m_feats_dir)[:youtube8m_cache_size]:
                t = feat_path.split('.npy')[0].split('#')
                video_id = t[0]
                index = '#' + t[1] + '.npy'
                ps.append(executor.submit(self.get_youtube8m_feat, video_id, index))
            for feat_path in os.listdir(self.stft_feats_dir)[:stft_cache_size]:
                t = feat_path.split('.npy')[0].split('#')
                video_id = t[0][:len(t[0]) // 2]
                index = '#' + t[1] + '#' + t[2] + '.npy'
                ps.append(executor.submit(self.get_stft_feat, video_id, index))
            for p in tqdm.tqdm(ps, total=len(ps)):
                p.result()
        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        label = sample['label']
        b1 = sample['b1']
        b2 = sample['b2']
        video_id = sample['id']
        index = sample['index']
        ts = sample['ts']
        if self.use_cache:
            youtube8m_feat = [torch.from_numpy(self.get_youtube8m_feat(video_id, x)) for x in b1['youtube8m'] + b2['youtube8m']]
            youtube8m_feat = torch.stack(youtube8m_feat).type(torch.FloatTensor)
            stft_feat = [torch.from_numpy(self.get_stft_feat(video_id, x)) for x in b1['stft'] + b2['stft']]
            stft_feat = torch.stack(stft_feat).type(torch.FloatTensor)
        else:
            youtube8m_feat = [torch.from_numpy(self.get_youtube8m_feat_without_cache(video_id, x)) for x in b1['youtube8m'] + b2['youtube8m']]
            youtube8m_feat = torch.stack(youtube8m_feat).type(torch.FloatTensor)
            stft_feat = [torch.from_numpy(self.get_stft_feat_without_cache(video_id, x)) for x in b1['stft'] + b2['stft']]
            stft_feat = torch.stack(stft_feat).type(torch.FloatTensor)
        return youtube8m_feat, stft_feat, label, video_id, index, ts

    def get_youtube8m_feat_without_cache(self, video_id, index):
        feat_path = os.path.join(self.youtube8m_feats_dir, '{}{}'.format(video_id, index))
        if os.path.exists(feat_path):
            return np.load(feat_path)
        else:
            print('{} not exist.'.format(feat_path))
            return np.zeros(1024)

    def get_stft_feat_without_cache(self, video_id, index):
        feat_path = os.path.join(self.stft_feats_dir, '{}{}{}'.format(video_id, video_id, index))
        if os.path.exists(feat_path):
            try:
                return np.load(feat_path)
            except:
                print('load {} failed'.format(feat_path))
                return np.zeros((257, 90))
        else:
            print('{} not exist.'.format(feat_path))
            return np.zeros((257, 90))

    @cached(cache=LRUCache(maxsize=youtube8m_cache_size))
    def get_youtube8m_feat(self, video_id, index):
        return self.get_youtube8m_feat_without_cache(video_id, index)

    @cached(cache=LRUCache(maxsize=stft_cache_size))
    def get_stft_feat(self, video_id, index):
        return self.get_stft_feat_without_cache(video_id, index)
