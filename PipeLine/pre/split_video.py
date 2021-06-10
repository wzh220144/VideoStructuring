import os
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
import cv2
import tqdm

def read_video_info(video_file):
    cap = cv2.VideoCapture(video_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return fps

def _split_video(video_path, target_path, t1, t2):
    res = os.system('ffmpeg -y -i {} -strict -2 -ss {:.2f} -to {:.2f} {}'.format(video_path, t1, t2, target_path))
    if res != 0:
        print('gen {} failed.'.format(target_path))

def split_video(video_dir, result_dir, split_dir, postfix, args):
    video_dir = os.path.join(video_dir, postfix)
    os.makedirs(os.path.join(split_dir, postfix), exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.max_worker) as executor:
        ps = []
        with open(os.path.join(result_dir, postfix), 'r') as fs:
            obj = json.load(fs)
            for key, value in obj.items():
                video_path = os.path.join(video_dir, key)
                fps = int(read_video_info(video_path))
                video_id = key.split('.')[0]
                for annotation in value['annotations']:
                    segment = annotation['segment']
                    target_path = os.path.join(split_dir, postfix, '{}#{:.2f}#{:.2f}#{}.mp4'.format(video_id, segment[0], segment[1], fps))
                    if segment[0] == segment[1]:
                        continue
                    cap = cv2.VideoCapture(target_path)
                    if not os.path.exists(target_path) or not cap.isOpened():
                        ps.append(executor.submit(_split_video, video_path, target_path, segment[0], segment[1]))
        for p in tqdm.tqdm(ps, total=len(ps)):
            p.result()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type = str, default = '/home/tione/notebook/VideoStructuring/dataset/videos')
    parser.add_argument('--result_dir', type = str, default = '/home/tione/notebook/VideoStructuring/dataset/result/seg')
    parser.add_argument('--train_postfix', type = str, default = 'train_5k_A')
    parser.add_argument('--test_postfix', type = str, default = 'test_5k_A')
    parser.add_argument('--split_dir', type = str, default = '/home/tione/notebook/VideoStructuring/dataset/split')
    parser.add_argument('--split_feats_dir', type = str, default = '/home/tione/notebook/VideoStructuring/dataset/split_feats')
    parser.add_argument('--max_worker', type = int, default = 10)
    args = parser.parse_args()
    #split_video(args.video_dir, args.result_dir, args.train_postfix, args)
    split_video(args.video_dir, args.result_dir, args.split_dir, args.test_postfix, args)

if __name__ == '__main__':
    main()
