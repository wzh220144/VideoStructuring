#encoding: utf-8
import sys, os
sys.path.append(os.getcwd())

import argparse
import tqdm
import glob
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from feats_extract.multimodal_feature_extract import MultiModalFeatureExtract
import time
import tensorflow as tf
import random


def process_file(gen, file_path, split_dir, fps, youtube8m_dir, resnet50_dir, vggish_dir, stft_dir, ocr_dir, asr_dir):
    if not os.path.exists(file_path):
        return
    try:
        gen.extract_feat(file_path, split_dir, fps, youtube8m_dir, resnet50_dir, vggish_dir, stft_dir, ocr_dir, asr_dir, True)
    except Exception as e:
        print(file_path, traceback.format_exc())

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--files_dir', default='/home/tione/notebook/dataset/videos/train_5k_A', type=str)
    parser.add_argument('--postfix', default='mp4', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--feat_dir', default='/home/tione/notebook/dataset/feats/train_5k_A')
    parser.add_argument('--split_dir', default='/home/tione/notebook/dataset/split/train_5k_A')
    parser.add_argument('--extract_youtube8m', type=bool, default=True)
    parser.add_argument('--extract_resnet50', type=bool, default=False)
    parser.add_argument('--extract_vggish', type=bool, default=False)
    parser.add_argument('--extract_stft', type=bool, default=True)
    parser.add_argument('--extract_asr', type=bool, default=False)
    parser.add_argument('--extract_ocr', type=bool, default=False)
    parser.add_argument('--max_worker', type=int, default=30)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--device', type=str, default='0')
    args = parser.parse_args()

    youtube8m_folder = args.feat_dir + '/youtube8m'
    resnet50_folder = args.feat_dir + '/resnet50'
    vggish_folder = args.feat_dir + '/vggish'
    stft_folder = args.feat_dir + '/stft'
    ocr_folder = args.feat_dir + '/ocr'
    asr_folder = args.feat_dir + '/asr'
    os.makedirs(args.split_dir, exist_ok=True)
    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(youtube8m_folder, exist_ok=True)
    os.makedirs(resnet50_folder, exist_ok=True)
    os.makedirs(vggish_folder, exist_ok=True)
    os.makedirs(stft_folder, exist_ok=True)
    os.makedirs(ocr_folder, exist_ok=True)
    os.makedirs(asr_folder, exist_ok=True)

    gens = []
    os.environ["CUDA_VISIBLE_DEVICES"]=args.device
    for device in ['0']:
        with tf.device('/gpu:{}'.format(device)):
            gens.append(MultiModalFeatureExtract(batch_size=args.batch_size,
                                                 extract_youtube8m=args.extract_youtube8m,
                                                 extract_resnet50 = args.extract_resnet50,
                                                 extract_vggish=args.extract_vggish,
                                                 extract_stft=args.extract_stft,
                                                 extract_ocr=args.extract_ocr,
                                                 extract_asr=args.extract_asr,
                                                 use_gpu=args.use_gpu,
                                                 device='cuda:{}'.format(device)))

    file_paths = glob.glob(args.files_dir+'/*.'+args.postfix)
    random.shuffle(file_paths)
    print('start extract feats')
    ps = []
    with ThreadPoolExecutor(max_workers=args.max_worker) as executor:
        for file_path in file_paths:
            vid = os.path.basename(file_path).split('.m')[0]
            youtube8m_dir = os.path.join(youtube8m_folder, vid)
            resnet50_dir = os.path.join(resnet50_folder, vid)
            vggish_dir = os.path.join(vggish_folder, vid)
            stft_dir = os.path.join(stft_folder, vid)
            ocr_dir = os.path.join(ocr_folder, vid)
            asr_dir = os.path.join(asr_folder, vid)
            t = random.randint(0, 0)
            ps.append(executor.submit(process_file, gens[t], file_path, args.split_dir, args.fps,
                                      youtube8m_dir, resnet50_dir,
                                      vggish_dir, stft_dir,
                                      ocr_dir, asr_dir))
            #process_file(gens[t], file_path, args.split_dir, args.fps, youtube8m_dir, resnet50_dir, vggish_dir, stft_dir, ocr_dir, asr_dir)
        for p in tqdm.tqdm(ps, total=len(ps), desc='feat extract'):
            p.result()
            end_time = time.time()
            #print('cur cost: {} sec'.format(end_time - start_time))
