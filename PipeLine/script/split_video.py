import os
import argparse

def split_video(video_path, t1, t2, target_path):
    if not os.path.exists(target_path):
        os.system('ffmpeg -y -i {} -strict -2 -ss {} -to {} {}'.format(video_path, t1, t2, target_path))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples_dir', type = str, default = '/home/tione/notebook/VideoStructuring/dataset/samples/seg')
    parser.add_argument('--window_size', type = int, default = 5)
    parser.add_argument('--mode', type = int, default = 1)
    parser.add_argument('--max_worker', type = int, default = 10)
    args = parser.parse_args()

if __name__ == '__main__':
    main()