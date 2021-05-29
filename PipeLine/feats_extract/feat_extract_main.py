#encoding: utf-8
import sys,os
sys.path.append(os.getcwd())

import argparse
import tqdm
import random
import glob
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from feats_extract.multimodal_feature_extract import MultiModalFeatureExtract
import time

def process_file(gen, file_path, video_path, audio_path, ocr_path, asr_path):
    if not os.path.exists(file_path):
        return
    try:
        print(file_path)
        gen.extract_feat(file_path, video_path, audio_path, ocr_path, asr_path, True)
    except Exception as e:
        print(file_path, traceback.format_exc())

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--files_dir', default='/home/tione/notebook/dataset/structuring/test5k_split_video', type=str)
    parser.add_argument('--postfix', default='mp4', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--feat_dir', default='/home/tione/notebook/dataset/structuring/test5k_split_video_feats')
    parser.add_argument('--extract_ocr', type=bool, default=False)
    parser.add_argument('--extract_video', type=bool, default=True)
    parser.add_argument('--extract_audio', type=bool, default=True)
    parser.add_argument('--extract_asr', type=bool, default=False)
    parser.add_argument('--max_worker', type=int, default=40)
    parser.add_argument('--use_gpu', type=bool, default=True)
    args = parser.parse_args()

    video_folder = args.feat_dir + '/video'
    audio_folder = args.feat_dir + '/audio'
    ocr_folder = args.feat_dir + '/ocr'
    asr_folder = args.feat_dir + '/asr'
    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(video_folder, exist_ok=True)
    os.makedirs(audio_folder, exist_ok=True)
    os.makedirs(ocr_folder, exist_ok=True)
    os.makedirs(asr_folder, exist_ok=True)

    gen = MultiModalFeatureExtract(batch_size = args.batch_size, extract_video = args.extract_video,
                                   extract_audio = args.extract_audio, extract_ocr = args.extract_ocr,
                                   extract_asr = args.extract_asr, use_gpu = args.use_gpu)

    file_paths = glob.glob(args.files_dir+'/*.'+args.postfix)
    random.shuffle(file_paths)
    print('start extract feats')
    ps = []
    with ThreadPoolExecutor(max_workers=args.max_worker) as executor:
        for file_path in file_paths:
            vid = os.path.basename(file_path).split('.m')[0]
            video_path = os.path.join(video_folder, vid+'.npy')
            audio_path = os.path.join(audio_folder, vid+'.npy')
            ocr_path = os.path.join(ocr_folder, vid+'.txt')
            asr_path = os.path.join(asr_folder, vid+'.txt')
            ps.append(executor.submit(process_file, gen, file_path, video_path, audio_path, ocr_path, asr_path))
            #process_file(gen, file_path, video_path, audio_path, ocr_path, asr_path)
        for p in tqdm.tqdm(ps, total=len(ps), desc='feat extract'):
            p.result()
            end_time = time.time()
            print('cur cost: {} sec'.format(end_time - start_time))
