#!/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2021.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File Name: lgss_moe.py
Author: Wang Zhihua <wangzhihua@.com>
Create Time: 2021/06/28 21:56:29
Brief:      
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np

class NeXtVLAD(nn.Module):
    def __init__(self, cfg, feature_size, frame, group, expansion, nextvlad_cluster_size):
        super(NeXtVLAD, self).__init__()
        self.cfg = cfg
        self.feature_size = feature_size
        self.frame = frame
        self.group = group
        self.expansion = expansion
        self.nextvlad_cluster_size = nextvlad_cluster_size
        self.fc1 = nn.Linear(feature_size, expansion * feature_size)
        self.attention = nn.Linear(expansion * feature_size, group)
        self.fc2 = nn.Linear(expansion * feature_size, group * nextvlad_cluster_size)
        self.bn1 = nn.BatchNorm1d(group * nextvlad_cluster_size)
        self.new_feature_size = expansion * feature_size // group
        self.cluster_weights = torch.empty(1, self.new_feature_size, nextvlad_cluster_size).cuda().to('cuda:0')
        nn.init.kaiming_normal_(self.cluster_weights, mode='fan_out', nonlinearity='relu')
        self.cluster_weights = torch.nn.parameter.Parameter(self.cluster_weights)
        self.bn2 = nn.BatchNorm1d(self.nextvlad_cluster_size * self.new_feature_size)

    def forward(self, x):
        x = self.fc1(x)
        attention = F.sigmoid(self.attention(x)).view(-1, self.frame * self.group, 1)
        reshape_x = x.view(-1, self.expansion * self.feature_size)
        t1 = self.fc2(reshape_x)
        t2 = self.bn1(t1)
        activation = t2.view(-1, self.frame * self.group, self.nextvlad_cluster_size)
        activation = F.softmax(activation, -1)
        activation = torch.multiply(activation, attention)
        a_sum = torch.sum(activation, dim=-2, keepdim = True)
        a = torch.multiply(a_sum, self.cluster_weights)
        activation = activation.permute(0, 2, 1)
        reshaped_x = x.view(-1, self.frame * self.group, self.new_feature_size)
        vlad = torch.matmul(activation, reshaped_x).permute(0, 2, 1)
        vlad = torch.sub(vlad, a)
        vlad = F.normalize(vlad, 2, 1)
        vlad = vlad.contiguous().view(-1, self.nextvlad_cluster_size * self.new_feature_size)
        vlad = self.bn2(vlad)
        #return vlad.view(-1, self.cfg.seq_len, self.nextvlad_cluster_size)
        return vlad

class AudNet(nn.Module):
    def __init__(self, cfg):
        super(AudNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3,3), stride=(2,1), padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=(1,3))

        self.conv2 = nn.Conv2d(64, 192, kernel_size=(3,3), stride=(2,1), padding=0)
        self.bn2 = nn.BatchNorm2d(192)
        self.relu2 = nn.ReLU(inplace=True)
        # self.pool2 = nn.MaxPool2d(kernel_size=(1,3))

        self.conv3 = nn.Conv2d(192, 384, kernel_size=(3,3), stride=(2,1), padding=0)
        self.bn3 = nn.BatchNorm2d(384)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(384, 256, kernel_size=(3,3), stride=(2,2), padding=0)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(256, 256, kernel_size=(3,3), stride=(2,2), padding=0)
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=(2,2))

        self.conv6 = nn.Conv2d(256, 512, kernel_size=(3,2), padding=0)
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(512, 512)

    def forward(self, x):  # [bs,1,257,90]
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.pool5(self.relu5(self.bn5(self.conv5(x))))
        x = self.relu6(self.bn6(self.conv6(x)))
        x = x.squeeze()
        out = self.fc(x)
        return out


class Cos(nn.Module):
    def __init__(self, cfg, config):
        super(Cos, self).__init__()
        self.shot_num = cfg.shot_num
        self.channel = config.sim_channel
        self.conv1 = nn.Conv2d(1, self.channel, kernel_size=(self.shot_num//2, 1))

    def forward(self, x):  # [batch_size, seq_len, shot_num, feat_dim]
        x = x.view(-1, 1, x.shape[2], x.shape[3])
        part1, part2 = torch.split(x, [self.shot_num//2]*2, dim=2)
        # batch_size*seq_len, 1, [self.shot_num//2], feat_dim
        part1 = self.conv1(part1).squeeze()
        part2 = self.conv1(part2).squeeze()
        x = F.cosine_similarity(part1, part2, dim=2)  # batch_size,channel
        return x

class BNet(nn.Module):
    def __init__(self, cfg, config):
        super(BNet, self).__init__()
        self.shot_num = cfg.shot_num
        self.channel = config.sim_channel
        self.conv1 = nn.Conv2d(1, self.channel, kernel_size=(cfg.shot_num, 1))
        self.max3d = nn.MaxPool3d(kernel_size=(self.channel, 1, 1))
        self.cos = Cos(cfg, config)

    def forward(self, x):  # [batch_size, seq_len, shot_num, feat_dim]
        context = x.view(x.shape[0]*x.shape[1], 1, -1, x.shape[-1])
        context = self.conv1(context)  # batch_size*seq_len,512,1,feat_dim
        context = self.max3d(context)  # batch_size*seq_len,1,1,feat_dim
        context = context.squeeze()
        sim = self.cos(x)
        bound = torch.cat((context, sim), dim=1)
        return bound


class BNetAud(nn.Module):
    def __init__(self, cfg, config):
        super(BNetAud, self).__init__()
        self.shot_num = cfg.shot_num
        self.channel = config.sim_channel
        self.AudNet = AudNet(cfg)
        self.conv1 = nn.Conv2d(1, self.channel, kernel_size=(cfg.shot_num, 1))
        self.conv2 = nn.Conv2d(1, self.channel, kernel_size=(cfg.shot_num//2, 1))
        self.max3d = nn.MaxPool3d(kernel_size=(self.channel, 1, 1))

    def forward(self, x):  # [batch_size, seq_len, shot_num, 257, 90]
        context = x.view(
            x.shape[0]*x.shape[1]*x.shape[2], 1, x.shape[-2], x.shape[-1])
        context = self.AudNet(context).view(
            x.shape[0]*x.shape[1], 1, self.shot_num, -1)
        part1, part2 = torch.split(context, [self.shot_num//2]*2, dim=2)
        part1 = self.conv2(part1).squeeze()
        part2 = self.conv2(part2).squeeze()
        sim = F.cosine_similarity(part1, part2, dim=2)
        bound = sim
        return bound


class LGSSone(nn.Module):
    def __init__(self, cfg, modal, config):
        super(LGSSone, self).__init__()
        self.mode = mode
        self.seq_len = cfg.seq_len

        self.num_layers = config.num_layers
        self.lstm_hidden_size = config.lstm_hidden_size
        if modal == 'aud':
            self.bnet = BNetAud(cfg, config)
            self.input_dim = config.input_dim
        else:
            self.bnet = BNet(cfg, config)
            self.input_dim = (config.input_dim + config.sim_channel)
        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            bidirectional=config.bidirectional)

        if config.bidirectional:
            self.output_dim = self.lstm_hidden_size * 2
        else:
            self.output_dim = self.lstm_hidden_size
        self.cfg = cfg

    def forward(self, x):
        self.lstm.flatten_parameters()
        if self.mode == 'place':
            x = self.next_vlad(x)
            x = self.fc(x).view(-1, self.cfg.seq_len, self.cfg.shot_num, self.cfg.model.place.output_dim)
        
        x = self.bnet(x)
        x = x.view(-1, self.seq_len, x.shape[-1])
        out, (_, _) = self.lstm(x, None)
        return out

class LGSS(nn.Module):
    def __init__(self, cfg):
        super(LGSS, self).__init__()
        self.cfg = cfg
        self.seq_len = cfg.seq_len
        self.modals = []
        self.se_dict = nn.ModuleDict()
        self.classify_dict = nn.ModuleDict()
        self.bnet_dict = nn.ModuleDict()
        self.fusion_dim = 0
        self.lgssone_dict = nn.ModuleDict()
        self.linear_dict = nn.ModuleDict()
        for modal, config in cfg.model.modal.items():
            if modal == 'fusion':
                continue
            if config.use != 1:
                continue
            self.modals.append(modal)
            self.lgssone_dict[modal] = LGSSone(cfg, modal, config)
            if self.cfg.model.model_mode == 1:
                self.linear_dict[modal] = nn.Linear(self.lgssone_dict[modal].output_dim, config.output_dim)    #线性层
            elif self.cfg.model.model_mode == 2:
                self.se_dict[modal] = SELayer(cfg, config.dropout_ratio, self.lgssone_dict[modal].output_dim, config.se_dim, config.reduction, config.output_dim)  #se层
            self.fusion_dim += config.output_dim
            self.classify_dict[modal] = nn.Linear(self.lgssone_dict[modal].output_dim, 2)
        fusion_config = cfg.model.modal['fusion']
        if self.cfg.model.model_mode == 1:
            self.linear_dict['fusion'] = nn.Linear(self.fusion_dim, fusion_config.output_dim)
        elif self.cfg.model.model_mode == 2:
            self.se_dict['fusion'] = SELayer(cfg, fusion_config.dropout_ratio, self.fusion_dim, config.se_dim, config.reduction, config.output_dim)
        self.classify_dict['fusion'] = nn.Linear(fusion_config.output_dim, 2)

    def forward(self, feat_dict):
        outs = {}
        if self.cfg.model.model_mode == 1:  #简单线性加权融合
            fusion = []
            for modal in self.modals:
                x = self.lgssone_dict[modal](feat_dict[modal + '_feats'])
                x = F.relu(x)
                #是否增加bn
                x = self.linear_dict[modal](x)
                fusion.append(x)
                x = self.classify_dict[modal](x)
                outs['{}_logits'.format(modal)] = x.view(-1, 2)
            fusion = torch.cat(fusion, dim = 2)
            fusion = self.linear_dict['fusion'](fusion)
            fusion = self.classify_dict['fusion'](fusion)
            outs['fusion_logits'] = fusion.view(-1, 2)
        elif self.cfg.model.model_mode == 2:    #se融合
            fusion = []
            for modal in self.modals:
                x = self.lgssone_dict[modal](feat_dict[modal + '_feats'])
                x = F.relu(x)
                #是否增加bn
                x = self.se_dict[modal](x)
                fusion.append(x)
                x = self.classify_dict[modal](x)
                outs['{}_logits'.format(modal)] = x.view(-1, 2)
            fusion = torch.cat(fusion, dim = 2)
            fusion = self.se_dict['fusion'](fusion)
            fusion = self.classify_dict['fusion'](fusion)
            outs['fusion_logits'] = fusion.view(-1, 2)

        for modal in self.modals + ['fusion']:
            x = outs['{}_logits'.format(modal)]
        return outs

class SELayer(nn.Module):
    def __init__(self, cfg, dropout_ratio, input_dim, se_dim, reduction, output_dim):
        super(SELayer, self).__init__()
        self.dropout_ratio = dropout_ratio
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.input_dim = input_dim
        self.fc1 = nn.Linear(self.input_dim, se_dim)
        self.bn1 = nn.BatchNorm1d(cfg.seq_len)
        self.fc2 = nn.Linear(se_dim, se_dim // reduction)
        self.bn2 = nn.BatchNorm1d(cfg.seq_len)
        self.fc3 = nn.Linear(se_dim // reduction, se_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc4 = nn.Linear(se_dim, output_dim)

    def forward(self, x):
        if self.dropout_ratio > 0:
            x = self.dropout(x)
        x = self.fc1(x)
        out = self.bn1(x)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        out = self.fc4(x * out)
        return out

if __name__ == '__main__':
    from mmcv import Config
    cfg = Config.fromfile("/home/tione/notebook/VideoStructuring/SceneSeg/lgss/config/train_transnet_v2.py")
    model = LGSS(cfg)
    place_feat = torch.randn(cfg.batch_size, cfg.seq_len, cfg.shot_num, 2048)
    vit_feat = torch.randn(cfg.batch_size, cfg.seq_len, cfg.shot_num, 768)
    act_feat = torch.randn(cfg.batch_size, cfg.seq_len, cfg.shot_num, 512)
    aud_feat = torch.randn(cfg.batch_size, cfg.seq_len, cfg.shot_num, 257, 90)
    target = torch.randn(cfg.batch_size, cfg.seq_len, cfg.shot_num, 1)
    print(target.shape, target.view(-1).shape)
    output = model(place_feat, vit_feat, act_feat, aud_feat)
    print(output.shape)
    print(cfg.batch_size)
    print(output.data.size())
