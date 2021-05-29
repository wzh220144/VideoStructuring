#coding=utf-8

import os
import argparse
import csv
import json
import random
import cv2
import glob
import tqdm

def read_video_info(video_path):
    """获取视频相关信息"""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h,w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frame_count,fps, h,w

def parse_annotation(input_annotation, video_dir):
    """
    输出Scene边界的Timestamp
    """
    with open(input_annotation) as f:
#         annotation_dict = json.load(f)
        annotation_dict = json.loads(f.read())
    scene_dict = {}
    for key, value in tqdm.tqdm(annotation_dict.items()):
        video_id = key.strip().split('.mp4')[0]
        frame_count, fps, h,w = read_video_info(os.path.join(video_dir, key))
        annotations= value["annotations"]
        boundary_list = [annotations[0]["segment"][0]*fps]
        if boundary_list[0] > 0.0:
            print("{} start at non-zero, with time: {} sec".format(key, boundary_list[0]))
            continue 
        for i in range(len(annotations)):
            if not annotations[i]["labels"]:
                print("{} at time {} without labels".format(key, annotations[i]["segment"]))
                continue
            boundary_list.append(annotations[i]["segment"][1]*fps)
        scene_dict[video_id] = boundary_list[1:-1]
        #assert len(scene_dict[video_id])>0, "{}:{}".format(key, value)
    return scene_dict

def match_shot_scene_boundary(save_dir, shot_dict, scene_dict):
    """
    将与场景边界最近邻的shot boundary认为是转场
    """
    for video_id, scene_list in scene_dict.items():
        positive_shots = set()
        if len(shot_dict.get(video_id, [])) == 0:
            print("video id {} not in shot_dict".format(video_id))
            continue
        shots = shot_dict[video_id]
        for scene in scene_list:
             positive_shots.add(shots.index(min(shots, key = lambda x: abs(x-scene))))

        with open(os.path.join(save_dir, video_id+".txt"), "w") as f:
            for i in range(len(shots)):
                if i in positive_shots:
                    f.write("{} 1\n".format(str(i).zfill(4)))
                else:
                    f.write("{} 0\n".format(str(i).zfill(4)))
             
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type = str, default = "../data/train799/video")
    parser.add_argument('--data_root', type = str, default = "../data/train799")
    parser.add_argument('--input_annotation', type = str, default = "../../dataset/gt_json/train799.json")
    args = parser.parse_args()
    print(args)

    shot_dict = {}
    for video_file in glob.glob(args.video_dir + "/*.mp4"):
        video_id = video_file.split('/')[-1].split(".mp4")[0]
        video_stats_path = os.path.join(args.data_root, "shot_stats", video_id+".csv")
        video_shot_txt = os.path.join(args.data_root, "shot_txt", video_id+".txt")
        if not os.path.exists(os.path.join(args.data_root, "shot_split_video", video_id)):
            print("{} not exists".format(os.path.join(args.data_root, "shot_split_video", video_id)))
            continue
        shot_frame_list = []
        with open(video_shot_txt) as f:
            for line in f:
                shot_frame_list.append(int(line.strip().split()[1]))

        if not shot_frame_list:
            print("file {} is empty".format(video_shot_txt))
            continue
        shot_dict[video_id] = shot_frame_list[:-1] 
    
      
    scene_dict = parse_annotation(args.input_annotation, args.video_dir)
    print('shot_dict num', len(shot_dict))
    print('scene_dict num', len(scene_dict))
    
    save_label_dir = os.path.join(args.data_root, "labels")
    os.makedirs(save_label_dir, exist_ok = True)
    match_shot_scene_boundary(save_label_dir, shot_dict, scene_dict)

    split_dict = {"train":[], "val":[], "test":[], "all":[]}
    for gt_txt in os.listdir(os.path.join(args.data_root, "labels")):
        video_id = gt_txt.split(".txt")[0]
        choice = random.randint(0,4)
        if choice <= 2:
            split_dict["train"].append(video_id)
        if choice == 3:
            split_dict["val"].append(video_id)
        if choice == 4:
            split_dict["test"].append(video_id)
        split_dict["all"].append(video_id)
    print("...split json done...")
    
    os.makedirs(os.path.join(args.data_root, "meta"), exist_ok = True)
    with open(os.path.join(args.data_root, "meta", "split.json"), "w") as f:
        json.dump(split_dict, f, indent = 4)

    for gt_txt in os.listdir(os.path.join(args.data_root, "labels")):
       video_id = gt_txt.split(".txt")[0]
       with open(os.path.join(args.data_root, "meta", "list_test.txt"), "a+") as f:
           f.write("".join([video_id, ".mp4", "\n"]))