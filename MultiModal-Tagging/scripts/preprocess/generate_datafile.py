import os
import time
import argparse
import sys
import tqdm
import random
import json

def convert_dataformat(line, out_file_dir, frame_npy_folder, audio_npy_folder, image_folder, text_txt_folder, modal_num):
  global count
  line = line.strip()
  if len(line.split('\t'))!=2: return
  path,label = line.split('\t')


  #skip item not in dict
  if len(tag_dict.keys()) > 0:
    tags = label.split(',')
    tag_in_dict = [tag for tag in tags if tag in tag_dict]
    if len(tag_in_dict)==0:
      print("{} do not match any tag in tag dict".format(tags))
      return
    label = ','.join(tag_in_dict)

  vid = os.path.basename(path).split('.m')[0]
  frame_npy_path = os.path.join(frame_npy_folder, vid+'.npy')
  audio_npy_path = os.path.join(audio_npy_folder, vid+'.npy')
  image_path = os.path.join(image_folder, vid+'.jpg')
  text_path = os.path.join(text_txt_folder, vid+'.txt')
  # text = open(text_path, 'r').read() if os.path.exists(text_path) else ""
  file_paths = [frame_npy_path, audio_npy_path, image_path, text_path] 
  exists_num = sum([os.path.exists(p) for p in file_paths])
  if exists_num < modal_num:
    print("WARNING: missing modal file: {}/{}".format(exists_num, modal_num))
    return
  try:
    for data_line in file_paths:
      if os.path.exists(data_line):
          out_file.write(data_line+'\n')
    out_file.write(label + '\n\n')
    count += 1
  except Exception as e:
        print(e)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--info_file', default='/home/tione/notebook/dataset/GroundTruth/structuring_tagging_info.txt',type=str)
    parser.add_argument('--out_file_dir', default='/home/tione/notebook/dataset/train_5k_A/tag_sample', type=str)
    parser.add_argument('--tag_dict_path', default='/home/tione/notebook/dataset/label_id.txt', type=str)
    parser.add_argument('--frame_npy_folder', default='/home/tione/notebook/dataset/train_5k_A/split_feats/video_npy', type=str)
    parser.add_argument('--audio_npy_folder', default='/home/tione/notebook/dataset/train_5k_A/split_feats/audio_npy', type=str)
    parser.add_argument('--text_txt_folder', default='/home/tione/notebook/dataset/train_5k_A/split_feats/ocr_txt',type=str) 
    parser.add_argument('--image_folder', default='/home/tione/notebook/dataset/train_5k_A/split_feats/image_jpg', type=str)
    parser.add_argument('--modal_num', default=4, type=int)
    parser.add_argument('--split_file', default='/home/tione/notebook/dataset/train_5k_A/shot_transnet_v2/meta/split.json', type=str)
    parser.add_argument('--ratio', default=0.9, type=float)
    count = 0
    args = parser.parse_args()
    if os.path.exists(args.tag_dict_path):
        with open(args.tag_dict_path, 'r') as f:
            tag_dict = {}
            for line in f:
                line = line.strip().split('\t')
                tag_dict[line[0]] = line[1]
            print(tag_dict)

    with open(args.split_file, 'r') as f:
        obj = json.load(f)
    train = set(obj['train'])
    val = set(obj['val'])

    with open(args.info_file,encoding='utf-8') as f:
        lines = [line for line in f]

        os.makedirs(args.out_file_dir, exist_ok=True)
        train_file = os.path.join(args.out_file_dir, 'train.txt')
        val_file = os.path.join(args.out_file_dir, 'val.txt')
        out_file_train = open(train_file,'w',encoding='utf-8')
        out_file_val = open(val_file,'w',encoding='utf-8')
        for index, line in tqdm.tqdm(enumerate(lines), total=len(lines)):
            if line.split('\t')[0].split('#')[0] in train:
                out_file = out_file_train
            else:
                out_file = out_file_val
            convert_dataformat(line, out_file, args.frame_npy_folder, args.audio_npy_folder, args.image_folder, args.text_txt_folder, args.modal_num)
    print(count)
