from __future__ import print_function, absolute_import
import argparse
import os
import os.path as osp
from collections import OrderedDict
from datetime import datetime
import multiprocessing
import numpy as np
import pickle
import pdb
from PIL import Image
import sys
import json
import time
from pprint import pprint

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import models
from torchvision import transforms


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import pdb
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import tqdm

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

os.environ["CUDA_VISIBLE_DEVICES"]='0,1'

resnet50_model_path = "/home/tione/notebook/VideoStructuring/pretrained/resnet50-19c8e357.pth"

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input
        # when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input
        # when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and
        # each residual block behaves like an identity.
        # This improves the model by 0.2~0.3%
        # according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().detach().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor

class ResNet50(torch.nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50, self).__init__()
        # self.base = models.resnet50(pretrained=pretrained)
        self.base = ResNet(Bottleneck, [3, 4, 6, 3])
        if pretrained:
            self.base.load_state_dict(torch.load(resnet50_model_path), strict=False)
 
    def forward(self, x):
        for name, module in self.base._modules.items():
            x = module(x)
            # print(name, x.size())
            if name == 'avgpool':
                x = x.view(x.size(0), -1)
                feature = x.clone()
        return feature, x

class Extractor(object):
    def __init__(self, model, args):
        super(Extractor, self).__init__()
        self.model = model
        self.use_gpu = args.use_gpu
        self.model.eval()
        # pprint(self.model.module)

    def extract_feature(self, data_loader):
        pre_feat_path = ''
        pre_num = -1
        features = []
        for imgs, feat_paths, nums in tqdm.tqdm(data_loader):
            if self.use_gpu == 1:
                imgs = imgs.cuda()
            outputs = self.model(imgs)
            for feat_path, num, feat, score in zip(feat_paths, nums, outputs[0], outputs[1]):
                if feat_path != pre_feat_path:
                    if len(features) > 0:
                        t = '/'.join(pre_feat_path.split('/')[:-1])
                        os.makedirs(t, exist_ok=True)
                        np.save(pre_feat_path, np.stack(features, axis = 0))
                    features = []
                    pre_num = -1
                if num != pre_num:
                    features.append(to_numpy(feat.cpu()))
                pre_num = num
                pre_feat_path = feat_path
        if len(features) > 0:
            np.save(pre_feat_path, np.stack(features, axis = 0))

class Preprocessor(object):
    def __init__(self, args, dataset, default_size, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.transform = transform
        self.default_size = default_size
        self.args = args

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        t = self.dataset[index]
        feat_path = t[1]
        img_path = t[0]
        num = int(img_path.split('_')[-1].split('.')[0])
        if os.path.isfile(img_path):
            img = Image.open(img_path).convert('RGB')
        else:
            img = Image.new('RGB', self.default_size)
            print('No such images: {:s}'.format(fpath))
        if self.transform is not None:
            img = self.transform(img)
        return img, feat_path, num

def get_data_loader(dataset, batch_size, workers, keyf_num, args):
    t = []
    dataset = sorted(dataset)
    for x in dataset:
        cols = x.split('/')
        vid = cols[-2]
        shotid = '_'.join(cols[-1].split('_')[:2])
        feat_path = os.path.join(args.feat_path, vid, shotid + '.npy')
        if os.path.exists(feat_path):
            continue
        num = int(cols[-1].split('_')[-1].split('.')[0])
        if num >= keyf_num:
            continue
        t.append([x, feat_path])
    dataset = t
    if len(dataset) % batch_size < batch_size:
        for i in range(batch_size - len(dataset) % batch_size):
            t.append(t[-1])

    normalizer = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    data_transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalizer,
    ])

    data_loader = DataLoader(
        Preprocessor(args, dataset, default_size=(256, 256), transform=data_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return data_loader

def run(dataset, extractor, args):
    data_loader = get_data_loader(dataset, args.batch_size, args.workers, args.keyf_num, args)
    extractor.extract_feature(data_loader)

def main(args):
    cudnn.benchmark = True
    model = ResNet50(pretrained=True)
    if args.use_gpu == 1:
        model = torch.nn.DataParallel(model)
        model = model.cuda()
    extractor = Extractor(model, args)
    
    video_dir_list = [os.path.join(args.source_img_path, x) for x in os.listdir(args.source_img_path)]
    dataset = []
    for video_dir in video_dir_list:
        dataset.extend([os.path.join(video_dir, x) for x in os.listdir(video_dir)])
    run(dataset, extractor, args)
    
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    parser = argparse.ArgumentParser("Place feature using ResNet50 with ImageNet pretrain")
    parser.add_argument('--use_gpu', type=int, default=0)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--data_root', type=str, default="/home/tione/notebook/dataset/train_5k_A/shot_hsv")
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--keyf_num', type=int, default=4)
    args = parser.parse_args()
    args.list_file = None
    args.source_img_path = osp.join(args.data_root, 'shot_keyf')
    args.feat_path = osp.join(args.data_root, 'place_feat')
    os.makedirs(args.feat_path, exist_ok=True)
    main(args)
