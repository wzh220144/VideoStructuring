#encoding: utf-8
import sys, os
sys.path.append(os.getcwd())
import glob
import tensorflow.compat.v1 as tf
import numpy as np
import cv2
import argparse
import time
import traceback
import json
import utils.tokenization as tokenization
from utils.train_util import get_label_name_dict
from src.feats_extract.multimodal_feature_extract import MultiModalFeatureExtract
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import tqdm
import os.path as osp

#################Inference Utils#################
tokokenizer = tokenization.FullTokenizer(vocab_file='/home/tione/notebook/VideoStructuring/MultiModal-Tagging/pretrained/bert/chinese_L-12_H-768_A-12/vocab.txt')
class TaggingModel():
    def __init__(self, args):
        tag_id_file = args.tag_id_file
        model_pb = args.model_pb

        if tag_id_file is None:
            raise
        else:
            self.label_name_dict = get_label_name_dict(tag_id_file, None)

        if model_pb is None:
            raise
        else:
            config = tf.ConfigProto(allow_soft_placement=True)
            if args.use_gpu == 1:
                config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            meta_graph_def = tf.saved_model.loader.load(self.sess, [tf.saved_model.tag_constants.SERVING], model_pb)
            signature_def = meta_graph_def.signature_def
            self.signature = signature_def[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

        batch_size = args.video_feats_extractor_batch_size
        imgfeat_extractor = args.imgfeat_extractor
        self.feat_extractor = MultiModalFeatureExtract(batch_size=batch_size, imgfeat_extractor = imgfeat_extractor, 
                             extract_video = args.extract_video, extract_audio = args.extract_audio, extract_text = args.extract_text, extract_img = args.extract_img)

    def image_preprocess(self, image, rescale=224):
        #resize to 224 and normlize to 0-1, then perform f(x)= 2*(x-0.5)
        if isinstance(image, type(None)):
          print("WARNING: test input image is None")
          return np.zeros((rescale, rescale, 3))
        if image.shape[0] !=rescale:
          image = cv2.resize(image, (rescale, rescale))
        image = 2*(image/(np.max(image)+1e-10) - 0.5)
        return image

    def text_preprocess(self, txt,max_len=128):
        tokens = ['[CLS]'] + tokokenizer.tokenize(txt)
        ids = tokokenizer.convert_tokens_to_ids(tokens)
        ids = ids[:max_len]
        ids = ids + [0]*(max_len-len(ids))
        return ids

    def preprocess(self, feat_dict, max_frame_num=300):
        ret_dict = {}
        for feat_type in feat_dict:
            if feat_type=='video':
                if len(feat_dict['video']) == 0:
                    feats = np.empty(shape=(max_frame_num, 0))
                else:
                    feats = np.zeros((max_frame_num, len(feat_dict['video'][0])))
                    valid_num = min(max_frame_num, len(feat_dict['video']))
                    feats[:valid_num] = feat_dict['video']
            elif feat_type=='audio':
                if len(feat_dict['audio']) == 0:
                    feats = np.zeros(shape=(max_frame_num, 128))
                else:
                    feats = np.zeros((max_frame_num, len(feat_dict['audio'][0])))
                    valid_num = min(max_frame_num, len(feat_dict['audio']))
                    feats[:valid_num] = feat_dict['audio']
            elif feat_type=='text':
                feats = self.text_preprocess(feat_dict['text'], 128)
            elif feat_type == 'image':
                feats = self.image_preprocess(feat_dict['image'])
            else:
                raise
            ret_dict[feat_type] = feats
        return ret_dict


    def inference(self, test_file, args):
        prefix = test_file.split("/")[-1].split(".m")[0]
        frame_npy_path = args.feat_dir + '/video_npy/' + prefix + '.npy'
        audio_npy_path = args.feat_dir + '/audio_npy/' + prefix + '.npy'
        txt_file_path = args.feat_dir + '/text_txt/' + prefix + '.txt'
        image_jpg_path = args.feat_dir + '/image_jpg/' + prefix + '.jpg'
        with self.sess.as_default() as sess:
            start_time = time.time()
            feat_dict = self.feat_extractor.extract_feat(test_file,
                    frame_npy_path=frame_npy_path,
                    audio_npy_path=audio_npy_path,
                    txt_file_path=txt_file_path,
                    image_jpg_path=image_jpg_path,
                    save=args.save_feat)
            end_time = time.time()
            print("feature extract cost time: {} sec".format(end_time - start_time))

            feat_dict_preprocess = self.preprocess(feat_dict)
            feed_dict ={}
            
            # Get input tensor.
            for key in feat_dict:
                if key in self.signature.inputs:
                  feed_dict[self.signature.inputs[key].name] = [feat_dict_preprocess[key]]
                
            if 'video_frames_num' in self.signature.inputs:
                feed_dict[self.signature.inputs['video_frames_num'].name] = [len(feat_dict['video'])]
            if 'audio_frames_num' in self.signature.inputs:
                feed_dict[self.signature.inputs['audio_frames_num'].name] = [len(feat_dict['audio'])]
                
            # Get output tensor.
            class_indexes = self.signature.outputs['class_indexes'].name
            predictions = self.signature.outputs['predictions'].name
            #video_embedding = self.signature.outputs['video_embedding'].name #(Optional)
            
            start_time = time.time()
            class_indexes,predictions = sess.run([class_indexes,predictions], feed_dict)
            end_time = time.time()
            
            print("multi-modal tagging model forward cost time: {} sec".format(end_time - start_time))


            labels=[self.label_name_dict[index] for index in class_indexes[0]]
            scores = predictions[0]

        return labels, scores

def run(test_file, args, model):
    key = test_file.split('/')[-1].split('.m')[0]
    p = osp.join(args.output_base, key)
    video_id, segment_id, start_time, end_time, _ = test_file.split("/")[-1].split(".m")[0].split("#")
    if osp.exists(p):
        labels = []
        scores = []
        f = open(p, 'r')
        for x in f.readline().strip('\n').split('\t')[1:]:
            cols = x.split('##')
            label = cols[0]
            score = float(cols[1])
            labels.append(label)
            scores.append(score)
        cur_output = {"segment": [start_time, end_time], "labels": labels, "scores": ["%.2f" % score for score in scores]}
        #print('exist ' + test_file + ', ' + str(cur_output))
    else:
        try:
            labels, scores = model.inference(test_file, args)
            cur_output = {"segment": [start_time, end_time], "labels": labels[:args.top_k], "scores": ["%.2f" % scores[i] for i in range(args.top_k)]}
            video_id = video_id + '.mp4'
            scores = [scores[i] for i in range(args.top_k)]
            f = open(p, 'w')
            f.write("{}\t{}\n".format(video_id, "\t".join(["{}##{:.3f}".format(labels[i], scores[i]) for i in range(len(scores))])))
            f.close()
            print('no exist ' + test_file + ', ' + str(cur_output))
        except:
            print(test_file)
            print(traceback.format_exc())
            cur_output = {}
    '''
    try:
        labels, scores = model.inference(test_file, args)
        cur_output = {"segment": [start_time, end_time], "labels": labels[:args.top_k], "scores": ["%.2f" % scores[i] for i in range(args.top_k)]}
        video_id = video_id + '.mp4'
        scores = [scores[i] for i in range(args.top_k)]
        f = open(p, 'w')
        f.write("{}\t{}\n".format(video_id, "\t".join(["{}##{:.3f}".format(labels[i], scores[i]) for i in range(len(scores))])))
        f.close()
        print('no exist ' + test_file + ', ' + str(cur_output))
    except:
        print(test_file)
        print(traceback.format_exc())
        cur_output = {}
    '''

    video_id, segment_id, start_time, end_time, _ = test_file.split("/")[-1].split(".m")[0].split("#")

    return (video_id, cur_output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_pb', default='/home/tione/notebook/VideoStructuring/MultiModal-Tagging/checkpoints/structuring_train5k/export/step_1000_0.3844',type=str)
    parser.add_argument('--tag_id_file', default='/home/tione/notebook/dataset/label_id.txt')
    parser.add_argument('--test_dir', default='/home/tione/notebook/dataset/structuring/split/test_5k_A')
    parser.add_argument('--feat_dir', default='/home/tione/notebook/dataset/structuring/split_feats/test_5k_A')
    parser.add_argument('--postfix', default='.mp4', type=str, help='test file type')
    parser.add_argument('--extract_text', type=bool, default=True)
    parser.add_argument('--extract_video', type=bool, default=True)
    parser.add_argument('--extract_audio', type=bool, default=True)
    parser.add_argument('--extract_img', type=bool, default=True)
    parser.add_argument('--top_k', type=int, default=20)
    parser.add_argument('--output', default="results/result_for_vis.txt", type=str) #用于可视化文件
    parser.add_argument('--output_base', default="/home/tione/notebook/dataset/results/tag", type=str) #用于可视化文件
    parser.add_argument('--output_json', default="/home/tione/notebook/dataset/results/tag/outjson.txt", type=str) #用于模型精度评估
    parser.add_argument('--max_worker', type=int, default=20)
    parser.add_argument('--save_feat', type=bool, default=True)
    parser.add_argument('--use_gpu', type=int, default=1)
    parser.add_argument('--imgfeat_extractor', type=str, default='Youtube8M')
    parser.add_argument('--video_feats_extractor_batch_size', type=int, default=8)
    args = parser.parse_args()
    if args.use_gpu == 1:
        os.environ["CUDA_VISIBLE_DEVICES"]='0,1'
    
    model = TaggingModel(args)
    test_files = glob.glob(args.test_dir+'/*'+args.postfix)
    test_files.sort()    
    output_result = {}

    #如果选择特征保存,相关路径需先创建
    frame_npy_folder = args.feat_dir + '/video_npy'
    audio_npy_folder = args.feat_dir + '/audio_npy'
    text_txt_folder = args.feat_dir + '/text_txt'
    image_jpg_folder = args.feat_dir + '/image_jpg'
    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(frame_npy_folder, exist_ok=True)
    os.makedirs(audio_npy_folder, exist_ok=True)
    os.makedirs(text_txt_folder, exist_ok=True)
    os.makedirs(image_jpg_folder, exist_ok=True)
    os.makedirs(args.output_base, exist_ok=True)

    ps = []
    with ThreadPoolExecutor(max_workers=args.max_worker) as executor:
        for test_file in test_files:
            ps.append(executor.submit(run, test_file, args, model))
        cnt = 0
        for p in ps:
            cnt += 1
            video_id, cur_output = p.result()
            if len(cur_output) == 0:
                continue
            video_id = video_id + '.mp4'
            if video_id not in output_result:
                output_result[video_id] = {"result": [cur_output]}
            else:
                output_result[video_id]["result"].append(cur_output)
    
    for key in output_result:
        output_result[key]["result"].sort(key=lambda x: float(x["segment"][0]))    

    with open(args.output_json, 'w', encoding="utf-8") as f:
        json.dump(output_result, f, ensure_ascii=False, indent = 4)
