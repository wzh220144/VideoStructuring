#!/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2021.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File Name: a.py
Author: Wang Zhihua <wangzhihua@.com>
Create Time: 2021/06/28 20:11:14
Brief:      
"""

import sys
import os
import logging
import time
import json
import datetime
from lgss.utils import load_checkpoint

def load_model(model_path, use_best = True):
    print(load_checkpoint(model_path, 1))

load_model('/home/tione/notebook/dataset/train_5k_A/shot_transnet_v2/model_best.pth.tar.2')
