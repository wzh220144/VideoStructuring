#coding=utf-8
#Author: jefxiong@tencent.com
#--------------------------------
# 并行音视频分离
#-------------------------------

import os
from multiprocessing.dummy import Pool
import glob
import argparse
import tqdm

def download_worker(file_path):
    index, file_path, dst_dir = file_path
    audio_path = dst_dir + '/' + file_path.split('/')[-1].replace('.mp4', '.wav')
    if os.path.exists(audio_path) and os.path.getsize(audio_path):
      return
    command = 'ffmpeg -loglevel error -y -i {} {}'.format(file_path, audio_path)
    os.system(command)
    print("{}: {} convert done".format(index, audio_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir',default = 'dataset/videos/youxi', type=str)
    parser.add_argument('--audio_dir',default = 'dataset/audios/youxi', type=str)
    parser.add_argument('--pool_size',default = 8, type=int, help="进程池数")
    args = parser.parse_args()
    assert os.path.exists(args.video_dir)
    os.makedirs(args.audio_dir, exist_ok=True)

    pool = Pool(args.pool_size)
    mp4_files = glob.glob(args.video_dir+"/*.mp4")
    mp4_files = [("{}/{}".format(i, len(mp4_files)), f, args.audio_dir) for i, f in enumerate(mp4_files)]
    try:
        for _ in tqdm.tqdm(pool.imap_unordered(download_worker, mp4_files), total=len(mp4_files)):
          pass
    finally:
        pool.close()
        pool.join()