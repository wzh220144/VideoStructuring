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

class VitExtractor(object):
  def __init__(self):
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224')

  def extract_rgb_frame_features_list(self, frame_rgbs, count):
        inputs = self.feature_extractor(images=frame_rgbs, return_tensors="pt")
        outputs = self.model(**inputs)
        pooler_output = outputs.pooler_output
        return pooler_output.detach().numpy()

