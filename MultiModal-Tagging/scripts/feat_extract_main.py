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
import tensorflow as tf

from src.feats_extract.multimodal_feature_extract import MultiModalFeatureExtract
from src.feats_extract.txt_extractor.text_requests import VideoASR,VideoOCR,ImageOCR


def process_file(gen, file_path, video_npy_path, audio_npy_path, text_txt_path, image_jpg_path, asr_txt_path, ocr_txt_path):
    if not os.path.exists(file_path):
        return
    try:
        print(file_path)
        gen.extract_feat(file_path, video_npy_path, text_txt_path, audio_npy_path, image_jpg_path, ocr_txt_path, asr_txt_path, True)
    except Exception as e:
        print(file_path, traceback.format_exc())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--files_dir', default='/home/tione/notebook/dataset/split/train_5k_A', type=str)
    parser.add_argument('--postfix', default='mp4', type=str)
    parser.add_argument('--datafile_path', default='dataset/datafile.txt')
    parser.add_argument('--image_batch_size', default=20, type=int)
    parser.add_argument('--feat_dir', default='/home/tione/notebook/dataset/train_5k_A/split_feats')
    parser.add_argument('--extract_video', action="store_true", default=False)
    parser.add_argument('--extract_img', action="store_true", default=False)
    parser.add_argument('--extract_audio', action="store_true", default=False)
    parser.add_argument('--extract_asr', action="store_true", default=False)
    parser.add_argument('--extract_ocr', action="store_true", default=False)
    parser.add_argument('--max_worker', type=int, default=10)
    parser.add_argument('--use_gpu', action="store_true", default=False)
    parser.add_argument('--log_file', type=str, default='/home/tione/notebook/VideoStructuring/err.log')
    parser.add_argument('--cuda_devices', type=str, default='0,1')
    args = parser.parse_args()
    print(args)

    video_npy_folder = args.feat_dir + '/video_npy'
    img_jpg_folder = args.feat_dir + '/image_jpg'
    audio_npy_folder = args.feat_dir + '/audio_npy'
    text_txt_folder = args.feat_dir + '/text_txt'
    ocr_txt_folder = args.feat_dir + '/ocr_txt'
    asr_txt_folder = args.feat_dir + '/asr_txt'
    image_jpg_folder = args.feat_dir + '/image_jpg'
    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(video_npy_folder, exist_ok=True)
    os.makedirs(image_jpg_folder, exist_ok=True)
    os.makedirs(audio_npy_folder, exist_ok=True)
    os.makedirs(text_txt_folder, exist_ok=True)
    os.makedirs(ocr_txt_folder, exist_ok=True)
    os.makedirs(asr_txt_folder, exist_ok=True)
    os.makedirs(image_jpg_folder, exist_ok=True)

    asr_extractor = VideoASR(args.use_gpu, args.max_worker)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    gen = MultiModalFeatureExtract(batch_size = args.image_batch_size,
            extract_video = args.extract_video,
            extract_img = args.extract_img,
            extract_audio = args.extract_audio,
            extract_ocr = args.extract_ocr,
            extract_asr = args.extract_asr,
            use_gpu = args.use_gpu,
            asr_extractor = asr_extractor,
            log_file = args.log_file)
                
    file_paths = glob.glob(args.files_dir+'/*.'+args.postfix)

    print('start extract feats')
    ps = []
    with ThreadPoolExecutor(max_workers=args.max_worker) as executor:
        for file_path in file_paths:
            vid = os.path.basename(file_path).split('.m')[0]
            video_npy_path = os.path.join(video_npy_folder, vid+'.npy')
            audio_npy_path = os.path.join(audio_npy_folder, vid+'.npy')
            image_jpg_path = os.path.join(image_jpg_folder, vid+'.jpg')
            text_txt_path = os.path.join(text_txt_folder, vid+'.txt')
            asr_txt_path = os.path.join(asr_txt_folder, vid+'.txt')
            ocr_txt_path = os.path.join(ocr_txt_folder, vid+'.txt')
            ps.append(executor.submit(process_file, gen, file_path, video_npy_path, audio_npy_path, text_txt_path, image_jpg_path, asr_txt_path, ocr_txt_path))
            #process_file(file_path, video_npy_path, audio_npy_path, text_txt_path, image_jpg_path, asr_txt_path, ocr_txt_path)
        for p in tqdm.tqdm(ps, total=len(ps)):
            p.result()
