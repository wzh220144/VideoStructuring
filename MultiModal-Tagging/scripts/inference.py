#encoding: utf-8
import sys,os
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

os.environ["CUDA_VISIBLE_DEVICES"]='0'

#################Inference Utils#################
tokokenizer = tokenization.FullTokenizer(vocab_file='pretrained/bert/chinese_L-12_H-768_A-12/vocab.txt')
class TaggingModel():
    def __init__(self, configs):
        tag_id_file = configs.get('tag_id_file', None)
        model_pb = configs.get('model_pb', None)
        if tag_id_file is None:
            raise
        else:
            self.label_name_dict = get_label_name_dict(tag_id_file, None)
        if model_pb is None:
            raise
        else:
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            meta_graph_def = tf.saved_model.loader.load(self.sess, [tf.saved_model.tag_constants.SERVING], model_pb)
            signature_def = meta_graph_def.signature_def
            self.signature = signature_def[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

        batch_size = configs.get('video_feats_extractor_batch_size', 8)
        imgfeat_extractor = configs.get('imgfeat_extractor', 'Youtube8M')
        self.feat_extractor = MultiModalFeatureExtract(batch_size=batch_size, imgfeat_extractor= imgfeat_extractor, 
                             extract_video = True, extract_audio = True,extract_text = True)

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
                feats = np.zeros((max_frame_num, len(feat_dict['video'][0])))
                valid_num = min(max_frame_num, len(feat_dict['video']))
                feats[:valid_num] = feat_dict['video']
            elif feat_type=='audio':
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


    def inference(self, test_file):
        with self.sess.as_default() as sess:
            start_time = time.time()
            feat_dict = self.feat_extractor.extract_feat(test_file, save=False)
            end_time = time.time()
            print("feature extract cost time: {} sec".format(end_time -start_time))
            if 'text' in feat_dict:
                print(feat_dict['text'])

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
            
            print("multi-modal tagging model forward cost time: {} sec".format(end_time -start_time))


            labels=[self.label_name_dict[index] for index in class_indexes[0]]
            scores = predictions[0]

        return labels, scores


def parse_info_txt(gt_txt):
    valid_videos = set()
    with open(gt_txt, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.strip().startswith("dataset/video_npy/Youtube8M/deconstruction"):
                valid_videos.add("dataset/videos/deconstruction/" + line.strip().split("/")[-1].replace(".npy", ".mp4"))
    return valid_videos

#################Inference Utils#################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_pb', default='checkpoints/ds/export/step_10000_0.6879',type=str)
    parser.add_argument('--tag_id_file', default='dataset/dict/tag-id-deconstruction_b0.txt')
    parser.add_argument('--test_dir', default='dataset/videos/deconstruction')
    parser.add_argument('--postfix', default='.mp4', type=str, help='test file type')
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--output', default="test_val_res.txt", type=str)
    parser.add_argument('--output_json', default="output_json.json", type=str)
    args = parser.parse_args()
    
    configs={'tag_id_file': args.tag_id_file, 'model_pb': args.model_pb}
    model = TaggingModel(configs)
    valid_videos = parse_info_txt("dataset/datafile/val.txt")
    test_files = glob.glob(args.test_dir+'/*'+args.postfix)
    test_files.sort()    
    test_files = [test_file for test_file in test_files if test_file in valid_videos]
    output_result = {}

    for test_file in test_files[:3]:
        print(test_file)
        try:
          labels, scores = model.inference(test_file)
        except:
          print(traceback.format_exc())
          
        for i in range(len(labels)):
          print("{}:{:.3f}".format(labels[i], scores[i]))
        print("-"*100)
 
        if args.output_json is not None:
            print(args.output_json)
            cur_output = {}
            output_result[test_file.split("/")[-1]] = cur_output
            cur_output["result"] = [{"labels": labels[:args.top_k], "scores": ["%.2f" % scores[i] for i in range(args.top_k)]}]
            # output_json["labels"] = labels[:args.top_k]
            # output_json["scores"] = ["%.2f" % scores[i] for i in range(args.top_k)]
            print(cur_output)

        if args.output is not None:
            print(args.output)
            with open(args.output, 'a+') as f:
                video_id = test_file.split('/')[-1].split('.')[0]
                scores = [scores[i] for i in range(args.top_k)]
                f.write("{}\t{}\n".format(video_id, "\t".join(["{}##{:.3f}".format(labels[i], scores[i]) for i in range(len(scores))]))) 
    
    with open(args.output_json, 'a+', encoding="utf-8") as f:
        json.dump(output_result, f, ensure_ascii=False, indent = 4)
