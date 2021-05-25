#coding=utf-8
#Author: jefxiong@tencent.com
#--------------------------------
# 将json gt文件转换为模型训练数据格式
# json format: {'video_id': {'annotations': {'segment': {}, 'labels': {}}}}
#--------------------------------

import json
import glob
import os
import argparse
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def split_video(video_path, t1, t2, target_path):
    if not os.path.exists(target_path):
        os.system('ffmpeg -y -i {} -strict -2 -ss {} -to {} {}'.format(video_path, t1, t2, target_path))

def read_video_info(video_path):
    """获取视频相关信息"""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h,w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frame_count, fps, h, w
        
if __name__ == '__main__':
    """
    Usuage: 
            python scripts/json2info.py --split_video_dir ../dataset/train799-segment
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', default="../dataset/videos/train_5k", type= str)
    parser.add_argument('--json_path', default="../dataset/structuring/GourndTruth/structuring_tagging_info.txt", type = str)
    parser.add_argument('--save_path', default="../dataset/info/structuring_train5k.txt", type=str)
    parser.add_argument('--convert_type', default = 'structuring', type= str) # 'tagging' or 'structuring'
    parser.add_argument('--split_video_dir', default=None, type=None)    # only in structuring
    args = parser.parse_args()
    
   
    if args.split_video_dir is not None:
        os.makedirs(args.split_video_dir, exist_ok=True)
        
    load_json = json.loads(open(args.json_path, 'r').read())
    if args.convert_type == "tagging":
        save_file = open(args.save_path, 'w')
        for key in load_json:
            x = []
            for ann in load_json[key]["annotations"]:
                x.extend(ann['labels'])
            x = set(x)
            save_file.write("{}\t{}\n".format(key, ",".join(list(x))))
            save_file.flush()
    elif args.convert_type == "structuring":
        save_file = open(args.save_path, 'w')
        results = []
        with ThreadPoolExecutor(max_workers=32) as executor:
            for key in load_json:
                anns = load_json[key]['annotations']
                video_path = os.path.join(args.video_dir, key)
                frame_count, fps, h, w = read_video_info(video_path)
                for i in range(len(anns)):
                    segment = anns[i]['segment']
                    labels = ','.join(anns[i]['labels'])
                    target_name =  "{}#{:02d}#{:.3f}#{:.3f}#{}.mp4".format(key.split('.')[0], i, 
                                                                           segment[0], segment[1], int(np.ceil(fps)))
                    if args.split_video_dir:
                        target_path = os.path.join(args.split_video_dir, target_name)
                        results.append(executor.submit(split_video, video_path, segment[0], segment[1], target_path))
                    save_file.write("{}\t{}\n".format(target_name, labels))
                save_file.flush()
        if args.split_video_dir:
            results = [result.result() for result in results]
    else:
        print("convert_type should be tagging or structuring")
        raise
        
        

