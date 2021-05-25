import argparse
import os
from moviepy.editor import (VideoFileClip,TextClip,AudioClip,CompositeVideoClip)
import random
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import cv2

def parse_inference_file(inference_file):
  assert os.path.exists(inference_file)
  inference_list=[]
  with open(inference_file, 'r') as inference:
    for line in inference:
      items = line.strip().split('\t')
      items[1:] = [str(i+1)+'.'+items[i+1] for i in range(len(items[1:]))]
      video_path = items[0]
      tag_with_scores = "\n".join(items[1:])
      inference_list.append([video_path,tag_with_scores])
    return inference_list

def parse_gt_file(gt_file):
  gt_dict={}
  if gt_file:
    with open(gt_file, 'r') as f:
      for line in f:
        items = line.strip().split('\t')
        tags = items[1]
        key = items[0].replace('.mp4','').replace('.jpg', '')
        gt_dict[key] = str(tags)
  return gt_dict

def parse_tag_id_file(tag_id_file):
  tag_id_dict={}
  with open(tag_id_file, 'r') as f:
    for line in f:
      tag,idx = line.strip().split('\t')
      tag_id_dict[tag] = int(idx)
  return tag_id_dict

def write_video_file(file_path, pred_label_score, gt_info, save_dir):
    video_clip = VideoFileClip(file_path)
    text_clip = TextClip(txt=pred_label_score,
                         font='utils/SimHei.ttf',
                         color='white',
                         fontsize=32,
                         bg_color='black',
                         align='West').set_pos(("left","top")).set_duration(video_clip.duration)
    compose_list = [video_clip, text_clip]
    if gt_info!="":
      gt_text_clip = TextClip(txt=gt_info,
                              font='utils/SimHei.ttf',
                              color='white',
                              fontsize=32,
                              bg_color='black',
                              align='East').set_pos(("right", "bottom")).set_duration(video_clip.duration)
      compose_list.append(gt_text_clip)
    result = CompositeVideoClip(compose_list)
    video_name = os.path.basename(file_path)
    result.write_videofile(save_dir+"/"+video_name,
                           fps=25,
                           codec='libx264',
                           audio_codec='aac',
                           temp_audiofile='temp-audio.m4a',
                           remove_temp=True)

def write_image_file(file_path, pred_label_score, gt_info, save_dir):
    file_name = os.path.basename(file_path)
    image = cv2.imread(file_path, -1)
    font = ImageFont.truetype('utils/SimHei.ttf', 30)
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    draw.text((50, 50), pred_label_score, font=font, fill=(0, 0, 255))
    draw.text((image.shape[1]//2, image.shape[0]//2), gt_info, font=font, fill=(0, 0, 255))
    image = np.array(img_pil)
    cv2.imwrite(save_dir+"/"+file_name, image)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--test_dir", type=str, default="dataset/videos/deconstruction")
  parser.add_argument("--postfix", type=str, default='mp4')   
  parser.add_argument("--inference_file", type=str, default="test_train_res.txt")
  parser.add_argument("--gt_file", type=str, default="dataset/info/info.txt")
  parser.add_argument("--save_dir", type=str, default=None)
  parser.add_argument("--sample_num", type=int, default=50)
  parser.add_argument("--filter_tag_name", type=str, default=None)
  parser.add_argument("--tag_id_file", type=str, default= "dataset/dict/tag-id-deconstruction_b0.txt")
  args = parser.parse_args()
  os.makedirs(args.save_dir, exist_ok=True)

  inference_list = parse_inference_file(args.inference_file)
  gt_dict = parse_gt_file(args.gt_file)
  tag_id_dict = parse_tag_id_file(args.tag_id_file)

  sample_num = args.sample_num if args.sample_num > 0 else len(inference_list)
  idxs = list(range(len(inference_list)))
  random.shuffle(idxs)
  sample_index = 0
  for i in idxs:
    file_path = args.test_dir +'/'+inference_list[i][0]+'.'+args.postfix 
    gt_info = gt_dict.get("/cephfs/group/eg-qboss-video-commercialization/lorayliu/experiments/MultiModal-Tagging/dataset/videos/deconstruction/" + inference_list[i][0], "")
  
    if not os.path.exists(file_path):
      continue
    print(gt_info)
    
    #filter by tag name
    if gt_info !="" and args.filter_tag_name is not None:
      idx = tag_id_dict[args.filter_tag_name]
      same_tag = [tag for tag in tag_id_dict if tag_id_dict[tag]==idx] 
      gt_tags = gt_info.split('\n')[:-1]
      hit_tag = False
      for tag in same_tag:
        if tag in gt_tags:
          hit_tag = True
          break
      if hit_tag == False:
        continue

    if args.postfix in ['mp4']:
      write_video_file(file_path = file_path, 
                       pred_label_score=inference_list[i][1], 
                       gt_info=gt_info, 
                       save_dir = args.save_dir)
    elif args.postfix in ['jpg', 'png']:
      write_image_file(file_path = file_path,
                       pred_label_score=inference_list[i][1], 
                       gt_info=gt_info, 
                       save_dir = args.save_dir)
    else:
      raise

    sample_index = sample_index + 1
    if sample_index >= sample_num:
      break
