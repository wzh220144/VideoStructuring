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
    def __init__(self, cfg):
        super(Cos, self).__init__()
        self.shot_num = cfg.shot_num
        self.channel = cfg.model.sim_channel
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
    def __init__(self, cfg):
        super(BNet, self).__init__()
        self.shot_num = cfg.shot_num
        self.channel = cfg.model.sim_channel
        self.conv1 = nn.Conv2d(1, self.channel, kernel_size=(cfg.shot_num, 1))
        self.max3d = nn.MaxPool3d(kernel_size=(self.channel, 1, 1))
        self.cos = Cos(cfg)

    def forward(self, x):  # [batch_size, seq_len, shot_num, feat_dim]
        context = x.view(x.shape[0]*x.shape[1], 1, -1, x.shape[-1])
        context = self.conv1(context)  # batch_size*seq_len,512,1,feat_dim
        context = self.max3d(context)  # batch_size*seq_len,1,1,feat_dim
        context = context.squeeze()
        sim = self.cos(x)
        bound = torch.cat((context, sim), dim=1)
        return bound


class BNetAud(nn.Module):
    def __init__(self, cfg):
        super(BNetAud, self).__init__()
        self.shot_num = cfg.shot_num
        self.channel = cfg.model.sim_channel
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
    def __init__(self, cfg, mode="place"):
        super(LGSSone, self).__init__()
        self.seq_len = cfg.seq_len
        self.num_layers = cfg.model.num_layers
        self.lstm_hidden_size = cfg.model.lstm_hidden_size
        if mode == "place":
            self.input_dim = (cfg.model.place_feat_dim+cfg.model.sim_channel)
            self.bnet = BNet(cfg)
        elif mode == "vit":
            self.bnet = BNet(cfg)
            self.input_dim = (cfg.model.vit_feat_dim+cfg.model.sim_channel)
        elif mode == "act":
            self.bnet = BNet(cfg)
            self.input_dim = (cfg.model.act_feat_dim+cfg.model.sim_channel)
        elif mode == "aud":
            self.bnet = BNetAud(cfg)
            self.input_dim = cfg.model.aud_feat_dim
        else:
            pass
        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            bidirectional=cfg.model.bidirectional)

        if cfg.model.bidirectional:
            self.fc1 = nn.Linear(self.lstm_hidden_size * 2, 100)
        else:
            self.fc1 = nn.Linear(self.lstm_hidden_size, 100)
        self.fc2 = nn.Linear(100, 2)
        self.cfg = cfg

    def forward(self, x):
        self.lstm.flatten_parameters()
        x = self.bnet(x)
        x = x.view(-1, self.seq_len, x.shape[-1])
        out, (_, _) = self.lstm(x, None)
        out = F.relu(self.fc1(out))
        if self.cfg.model.model_mode == 1:
            out = self.fc2(out)
            out = out.view(-1, 2)
        elif self.cfg.model.model_mode == 2:
            pass
        return out
            
class LGSS(nn.Module):
    def __init__(self, cfg):
        super(LGSS, self).__init__()
        self.cfg = cfg
        self.seq_len = cfg.seq_len
        self.mode = cfg.dataset.mode
        self.num_layers = cfg.model.num_layers
        self.lstm_hidden_size = cfg.model.lstm_hidden_size
        self.ratio = cfg.model.ratio
        self.classify_head = {}
        self.dim = 0
        if 'place' in self.mode:
            self.bnet_place = LGSSone(cfg, "place")
            self.dim += 100
        if 'vit' in self.mode:
            self.bnet_vit = LGSSone(cfg, "vit")
            self.dim += 100
        if 'act' in self.mode:
            self.bnet_act = LGSSone(cfg, "act")
            self.dim += 100
        if 'aud' in self.mode:
            self.bnet_aud = LGSSone(cfg, "aud")
            self.dim += 100
        self.se_classify_head = nn.Linear(self.cfg.model.se_dim, 2)
        self.se_layer = SELayer(self.cfg, self.dim, self.cfg.model.reduction)

    def _classify(self, classify_layer, x):
        out = classify_layer(x)
        out = x.view(-1, 2)
        return out

    def forward(self, place_feat, vit_feat, act_feat, aud_feat):
        outs = {}
        if self.cfg.model.model_mode == 1:  #简单线性加权融合
            out = 0
            if 'place' in self.mode:
                place_bound = self.bnet_place(place_feat)
                out += self.ratio[0] * place_bound
                outs['place'] = place_bound
            if 'vit' in self.mode:
                vit_bound = self.bnet_vit(vit_feat)
                out += self.ratio[1] * vit_bound
                outs['vit'] = vit_bound
            if 'act' in self.mode:
                act_bound = self.bnet_act(act_feat)
                out += self.ratio[2] * act_bound
                outs['act'] = act_bound
            if 'aud' in self.mode:
                aud_bound = self.bnet_aud(aud_feat)
                out += self.ratio[3] * aud_bound
                outs['aud'] = aud_bound
            outs['fusion'] = out
        elif self.cfg.model.model_mode == 2:    #se融合
            inputs = []
            if 'place' in self.mode:
                place_bound = self.bnet_place(place_feat)
                inputs.append(place_bound)
            if 'vit' in self.mode:
                vit_bound = self.bnet_vit(vit_feat)
                inputs.append(vit_bound)
            if 'act' in self.mode:
                act_bound = self.bnet_act(act_feat)
                inputs.append(act_bound)
            if 'aud' in self.mode:
                aud_bound = self.bnet_aud(aud_feat)
                inputs.append(aud_bound)
            inputs = torch.concat(inputs, axis = 1)
            out = self.se_layer(inputs)
            out = self._classify(self.se_classify_head, out)
            outs['fusion'] = out
        return outs

class SELayer(nn.Module):
    def __init__(self, cfg, input_dim, reduction):
        super(SELayer, self).__init__()
        self.cfg = cfg
        if self.cfg.model.dropout_ratio > 0:
            self.dropout = nn.Dropout(p=self.cfg.model.dropout_ratio)
        self.input_dim = input_dim
        self.fc1 = nn.Linear(self.input_dim, self.cfg.model.se_dim)
        self.bn1 = nn.BatchNorm1d(self.cfg.model.se_dim)
        self.fc2 = nn.Linear(self.cfg.model.se_dim, self.cfg.model.se_dim // reduction)
        self.bn2 = nn.BatchNorm1d(self.cfg.model.se_dim // reduction)
        self.fc3 = nn.Linear(self.cfg.model.se_dim // reduction, self.cfg.model.se_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = x
        if self.cfg.dropout_ratio > 0:
            out = self.dropout(out)
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return x * out

if __name__ == '__main__':
    from mmcv import Config
    cfg = Config.fromfile("../config/train_hsv.py")
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
