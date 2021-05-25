#encoding: utf-8
import sys,os
sys.path.append(os.getcwd())

import time
import argparse
import tqdm
import random
import glob
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import os.path as osp

from src.feats_extract.multimodal_feature_extract import MultiModalFeatureExtract

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--files_dir', default='/home/tione/notebook/dataset/structuring/test5k_split_video', type=str)
    parser.add_argument('--postfix', default='mp4', type=str)
    parser.add_argument('--datafile_path', default='dataset/datafile.txt')
    parser.add_argument('--image_batch_size', default=32, type=int)
    parser.add_argument('--imgfeat_extractor', default='Youtube8M', type=str)
    parser.add_argument('--feat_dir', default='/home/tione/notebook/dataset/structuring/test5k_split_video_feats')
    parser.add_argument('--extract_text', type=bool, default=False)
    parser.add_argument('--extract_video', type=bool, default=True)
    parser.add_argument('--extract_audio', type=bool, default=True)
    parser.add_argument('--extract_img', type=bool, default=True)
    parser.add_argument('--max_worker', type=int, default=20)
    parser.add_argument('--use_gpu', type=bool, default=True)
    args = parser.parse_args()

    frame_npy_folder = args.feat_dir + '/video_npy'
    audio_npy_folder = args.feat_dir + '/audio_npy'
    text_txt_folder = args.feat_dir + '/text_txt'
    image_jpg_folder = args.feat_dir + '/image_jpg'
    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(frame_npy_folder, exist_ok=True)
    os.makedirs(audio_npy_folder, exist_ok=True)
    os.makedirs(text_txt_folder, exist_ok=True)
    os.makedirs(image_jpg_folder, exist_ok=True)

    gen =  MultiModalFeatureExtract(batch_size = args.image_batch_size,
                             imgfeat_extractor = args.imgfeat_extractor,
                             extract_video = args.extract_video,
                             extract_audio = args.extract_audio,
                             extract_text = args.extract_text,
                             extract_img = args.extract_img,
                             use_gpu = args.use_gpu)
    def process_file(file_path, frame_npy_path, audio_npy_path, text_txt_path, image_jpg_path):
      if not os.path.exists(file_path):
        return
      try:
          print(file_path)
          gen.extract_feat(file_path, frame_npy_path, audio_npy_path, text_txt_path, image_jpg_path, True)
      except Exception as e:
          print(file_path, traceback.format_exc())
            
    file_paths = glob.glob(args.files_dir+'/*.'+args.postfix)
    random.shuffle(file_paths)

    print('start extract feats')
    ps = []
    with ThreadPoolExecutor(max_workers=args.max_worker) as executor:
        for file_path in tqdm.tqdm(file_paths, total=len(file_paths)):
            vid = os.path.basename(file_path).split('.m')[0]
            frame_npy_path = os.path.join(frame_npy_folder, vid+'.npy')
            audio_npy_path = os.path.join(audio_npy_folder, vid+'.npy')
            image_jpg_path = os.path.join(image_jpg_folder, vid+'.jpg')
            text_txt_path = os.path.join(text_txt_folder, vid+'.txt')
            ps.append(executor.submit(process_file, file_path, frame_npy_path, audio_npy_path, text_txt_path, image_jpg_path))
        for p in ps:
            p.result()
