#!/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2021tencent.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File Name: vit_extractor.py
Author: Wang Zhihua <wangzhihua@tencent.com>
Create Time: 2021/06/23 00:57:12
Brief:      
"""

from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTModel
import torch.nn as nn
import torch

class VitExtractor(object):
  def __init__(self):
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('/home/tione/notebook/VideoStructuring/MultiModal-Tagging/pretrained/vit')
        self.model = ViTModel.from_pretrained('/home/tione/notebook/VideoStructuring/MultiModal-Tagging/pretrained/vit').cuda()
        self.model = nn.DataParallel(self.model)
        self.model = self.model.to('cuda:{}'.format(self.model.device_ids[0]))
        self.model.eval()

  def extract_rgb_frame_features_list(self, frame_rgbs, count):
        inputs = self.feature_extractor(images=frame_rgbs, return_tensors="pt")
        inputs = {k: v.to('cuda:{}'.format(self.model.device_ids[0])) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state[:,0,:]
        #last_hidden_state = outputs.pooler_output
        return last_hidden_state.detach().cpu().numpy()

