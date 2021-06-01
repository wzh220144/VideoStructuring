import torch
import os
import json
from torch.utils.data import DataLoader, Dataset
from cachetools import cached, LRUCache, TTLCache
import numpy as np

youtube8m_cache_size = 100000
stft_cache_size = 100000

class SegDataset(Dataset):
	def __init__(self, path, extract_youtube8m, extract_stft, postfix, feats_dir):
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

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, index):
		sample = self.samples[index]
		label = sample['label']
		b1 = sample['b1']
		b2 = sample['b2']
		video_id = sample['id']
		b1_youtube8m = [self.get_youtube8m_feat(video_id, x) for x in b1]
		b1_stft = [self.get_youtube8m_feat(video_id, x) for x in b1]
		b2_youtube8m = [self.get_youtube8m_feat(video_id, x) for x in b2]
		b2_stft = [self.get_youtube8m_feat(video_id, x) for x in b2]
		return b1_youtube8m, b1_stft, b2_youtube8m, b2_stft, label

	@cached(cache=LRUCache(maxsize=youtube8m_cache_size))
	def get_youtube8m_feat(self, video_id, index):
		feat_path = os.path.join(self.youtube8m_feats_dir, '{}{}'.format(video_id, index))
		if os.path.exists(feat_path):
			return np.load(feat_path)
		else:
			print('{} not exist.'.format(feat_path))
			return np.zeros(1024)

	@cached(cache=LRUCache(maxsize=stft_cache_size))
	def get_stft_feat(self, video_id, index):
		feat_path = os.path.join(self.stft_feats_dir, '{}{}'.format(video_id, index))
		if os.path.exists(feat_path):
			return np.load(feat_path)
		else:
			print('{} not exist.'.format(feat_path))
			return np.zeros((257, 90))
