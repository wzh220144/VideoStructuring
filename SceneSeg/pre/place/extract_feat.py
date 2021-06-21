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

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

resnet50_model_path = "/home/tione/notebook/VideoStructuring/pretrained/lgss/src/models/resnet50-19c8e357.pth"

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

'''
def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(resnet50_model_path), strict=False)
    return model
'''

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

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
    def __init__(self, model):
        super(Extractor, self).__init__()
        self.model = model
        # pprint(self.model.module)

    def extract_feature(self, data_loader, print_summary=True):
        self.model.eval()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        features = OrderedDict()
        scores = OrderedDict()

        end = time.time()
        for i, (imgs, fnames) in enumerate(data_loader):
            data_time.update(time.time() - end)
            outputs = self.model(imgs)

            for fname, feat, score in zip(fnames, outputs[0], outputs[1]):
                features[fname] = feat.cpu().data
                scores[fname] = score.cpu().data

            batch_time.update(time.time() - end)
            end = time.time()

            if print_summary:
                print('Extract Features: [{}/{}]\t'
                    'Time {:.3f} ({:.3f})\t'
                    'Data {:.3f} ({:.3f})\t'
                    .format(
                        i + 1, len(data_loader),
                        batch_time.val, batch_time.avg,
                        data_time.val, data_time.avg))
        return features, scores


class Preprocessor(object):
    def __init__(self, dataset, images_path, default_size, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.images_path = images_path
        self.transform = transform
        self.default_size = default_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname = self.dataset[index]
        fpath = osp.join(self.images_path, fname)
        if os.path.isfile(fpath):
            img = Image.open(fpath).convert('RGB')
        else:
            img = Image.new('RGB', self.default_size)
            print('No such images: {:s}'.format(fpath))
        if self.transform is not None:
            img = self.transform(img)
        return img, fname


def get_data(video_id, img_path, batch_size, workers):

    dataset = os.listdir(img_path)    # image nums
    if len(dataset) % batch_size < 8:   #保持完成batch size
        for i in range(8 - len(dataset) % batch_size):
            dataset.append(dataset[-1])

    # data transforms, Imagenet mean and std
    normalizer = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    data_transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalizer,
    ])

    # data loaders
    data_loader = DataLoader(
        Preprocessor(dataset, img_path, default_size=(256, 256), transform=data_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, data_loader


def get_img_folder(data_root, video_id):
    img_folder = osp.join(data_root, video_id)
    if osp.isdir(img_folder):
#         return os.path.join(img_folder, video_id)
        return img_folder
    else:
        print('No such movie: {}'.format(video_id))
        return None

def run(extractor, video_id, video_list, idx_m, args):
    print('****** {}, {} / {}, {} ******'.format(datetime.now(), idx_m+1, len(video_list), video_id))
    save_path = osp.join(args.save_path, video_id)
    os.makedirs(save_path, exist_ok=True)
    img_path = get_img_folder(args.source_img_path, video_id)    # video/image_path
    if not osp.isdir(img_path): #img没有生成, 先暂时return
        print('Cannot find images!')
        return

    feat_save_name = osp.join(save_path, 'feat.pkl')
    score_save_name = osp.join(save_path, 'score.pkl')
    if osp.isfile(feat_save_name) and osp.isfile(score_save_name):
        print('{}, {} exist.'.format(datetime.now(), video_id))
        return
    # create data loaders
    dataset, data_loader = get_data(video_id, img_path, args.batch_size, args.workers)

    # extract feature
    try:
        print('{}, extracting features...'.format(datetime.now()))
        feat_dict, score_dict = extractor.extract_feature(data_loader, print_summary=False)
        for key, item in feat_dict.items():
            item = to_numpy(item)
            os.makedirs(osp.join(args.save_feat_path,video_id),exist_ok = True)
            img_ind = key.split("_")[-1].split(".jpg")[0]
            if args.save_one_frame_feat:
                if img_ind is "0":
                    shot_ind = key.split("_")[1]
                    if not os.path.exists(osp.join(args.source_img_path,video_id,"shot_{}_img_1.jpg".format(shot_ind))):                                         
                        save_fn = osp.join(args.save_feat_path,video_id,"shot_{}.npy".format(shot_ind))
                        np.save(save_fn,item)
                                      
                if img_ind is "1":
                    shot_ind = key.split("_")[1]
                    save_fn = osp.join(args.save_feat_path,video_id,"shot_{}.npy".format(shot_ind))
                    np.save(save_fn,item)                    
                else:
                    continue
            else:
                save_fn = osp.join(args.save_feat_path,video_id,"{}.npy".format(key.split(".jpg")[0]))
                np.save(save_fn,item)
                    
        print('{}, saving...'.format(datetime.now()))
        with open(feat_save_name, 'wb') as f:
            pickle.dump(feat_dict, f)
        with open(score_save_name, 'wb') as f:
            pickle.dump(score_dict, f)
    except Exception as e:
        print('{} error! {}'.format(video_id, e))
    print('\n')

def main(args):
    print(args)
    cudnn.benchmark = True
    # create model
    model = ResNet50(pretrained=True)
    model = torch.nn.DataParallel(model)
    if args.use_gpu == 1:
        model = model.cuda()
    # create and extractor
    extractor = Extractor(model)
    
    if args.list_file is None:
        video_list = sorted(os.listdir(args.source_img_path))
    else:
        video_list = [x for x in open(args.list_file)] 
    video_list = [i.split(".m")[0] for i in video_list] ## to remove suffix .mp4 .mov etc. if applicable
    video_list = video_list[args.st:args.ed]
    print('****** Total {} videos ******'.format(len(video_list)))
    
    ps = []
    with ThreadPoolExecutor(args.max_worker) as executor:
        for idx_m, video_id in enumerate(video_list):
            ps.append(executor.submit(run, extractor, video_id, video_list, idx_m, args))
        for p in ps:
            p.result()
    '''
    for idx_m, video_id in enumerate(video_list):
        run(extractor, video_id, video_list, idx_m, args)
    '''


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Place feature using ResNet50 with ImageNet pretrain")
    parser.add_argument('--use_gpu', type=int, default=0)
    parser.add_argument('--max_worker', type=int, default=80)
    parser.add_argument('--data_root', type=str, default="/home/tione/notebook/dataset/videos/train_5k_A/shot_hsv")
    parser.add_argument('--save-one-frame-feat', type=bool, default=True)
    parser.add_argument('-b', '--batch-size', type=int, default=512)
    parser.add_argument('-j', '--workers', type=int, default=2)
    # parser.add_argument('--list_file', type=str, default=osp.join(data_root,'meta/list_test.txt'),
                        # help='The list of videos to be processed,\
                        # in the form of xxxx0.mp4\nxxxx1.mp4\nxxxx2.mp4\n \
                        #             or xxxx0\nxxxx1\nxxxx2\n')
    # parser.add_argument('--source_img_path', type=str,default=osp.join(data_root,'shot_keyf'))
    # parser.add_argument('--save_path',type=str,default=osp.join(data_root,'place_feat_raw'))
    # parser.add_argument('--save_feat_path',type=str,default=osp.join(data_root,'place_feat'))
    parser.add_argument('--st', type=int, default=0, help='start number') 
    parser.add_argument('--ed', type=int, default=9999999, help='end number')
    args = parser.parse_args()
    #args.list_file = osp.join(args.data_root,'meta/list_test.txt')
    args.list_file = None
    args.source_img_path = osp.join(args.data_root,'shot_keyf')
    args.save_path = osp.join(args.data_root,'place_feat_raw')
    args.save_feat_path = osp.join(args.data_root,'place_feat')
    main(args)
