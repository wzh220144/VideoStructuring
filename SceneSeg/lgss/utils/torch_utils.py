import torch
import shutil
import os
import os.path as osp
import pdb

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def save_checkpoint(state, is_best, epoch, save_path):
    os.makedirs(save_path, exist_ok=True)
    fpath = '{}/{}.pth.tar'.format(save_path, epoch)
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(save_path, 'model_best.pth.tar'))

def load_checkpoint(fpath, use_gpu=1):
    if osp.isfile(fpath):
        if use_gpu == 1:
            checkpoint = torch.load(fpath, map_location="cuda:0")
        else:
            checkpoint = torch.load(fpath, map_location='cpu')
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))

