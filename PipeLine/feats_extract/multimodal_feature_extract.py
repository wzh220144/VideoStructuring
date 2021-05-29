from __future__ import unicode_literals
import sys,os
import numpy as np
import cv2
import time
import tensorflow as tf
import json
import traceback
import random
from tqdm import tqdm

from feats_extract.imgfeat_extractor.youtube8M_extractor import YouTube8MFeatureExtractor
from feats_extract.imgfeat_extractor.finetuned_resnet101 import FinetunedResnet101Extractor
from feats_extract.txt_extractor.text_requests import VideoASR,VideoOCR
from feats_extract.audio_extractor import vggish_input,vggish_params,vggish_postprocess,vggish_slim

BASE = "/home/tione/notebook/VideoStructuring"
PCA_PARAMS_PATH = BASE + "/pretrained/vggfish/vggish_pca_params.npz"
VGGISH_CHECKPOINT_PATH = BASE + "/pretrained/vggfish/vggish_model.ckpt"
VIDEO_EXTRACTOR = 'Youtube8M'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

class MultiModalFeatureExtract(object):

    def get_video_extractor(self, device, batch_size):
        if VIDEO_EXTRACTOR == 'Youtube8M':
            return YouTube8MFeatureExtractor(device, use_batch=batch_size != 1)
        elif VIDEO_EXTRACTOR == 'FinetunedResnet101':
            return FinetunedResnet101Extractor()
        else:
            raise NotImplementedError(VIDEO_EXTRACTOR)

    """docstring for ClassName"""
    def __init__(self, batch_size = 1,
                 extract_youtube8m = True,
                 extract_vggish = True,
                 extract_ocr = True,
                 extract_asr = True,
                 use_gpu = True,
                 ):
        super(MultiModalFeatureExtract, self).__init__()
        self.extract_youtube8m = extract_youtube8m
        self.extract_vggish = extract_vggish
        self.extract_ocr = extract_ocr
        self.extract_asr = extract_asr
        self.batch_size = batch_size

        #视频特征抽取模型
        if extract_youtube8m:
            self.youtube8m_extractors = [
                    YouTube8MFeatureExtractor('cuda:0', use_batch = batch_size != 1),
                    YouTube8MFeatureExtractor('cuda:1', use_batch = batch_size != 1)
                    ]

        #音频特征抽取模型
        if extract_vggish:
            self.pproc = vggish_postprocess.Postprocessor(PCA_PARAMS_PATH)  # audio pca
            self.audio_graph = tf.Graph()
            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
            if use_gpu:
                config.gpu_options.allow_growth = True
            with self.audio_graph.as_default():
                self.audio_sess = tf.Session(graph=self.audio_graph, config=config)
                vggish_slim.define_vggish_slim(training=False)
                vggish_slim.load_vggish_slim_checkpoint(self.audio_sess, VGGISH_CHECKPOINT_PATH)
            self.features_tensor = self.audio_sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
            self.embedding_tensor = self.audio_sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)

        #OCR特征抽取模型
        if extract_ocr:
            self.ocr_extractor = VideoOCR(use_gpu)

        #ASR特征抽取模型
        if extract_asr:
            self.asr_extractor = VideoASR(use_gpu)

    def get_frames_same_interval(self, filename, every_ms, max_num_frames):
        video_capture = cv2.VideoCapture()
        if not video_capture.open(filename):
          print(sys.stderr, 'Error: Cannot open video file ' + filename)
          return
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        rate = int(video_capture.get(cv2.CAP_PROP_FPS))
        step = 1000.0 / rate
        frames = set([])
        cur_frame = 0
        frames.add(cur_frame)
        cur = 0
        index = 1
        while True:
            cur_frame += 1
            cur += step
            if cur_frame >= frame_count:
                break
            if cur >= index * every_ms:
                index += 1
                frames.add(cur_frame)
        frames.add(frame_count - 1)
        print('{} has {} frames, sample {} frames.'.format(filename, frame_count, len(frames)))

        cur_frame = 0
        frame_all = []
        cnt = 0
        while True:
            has_frames, frame = video_capture.read()
            if not has_frames:
                break
            if max_num_frames != -1 and cnt >= max_num_frames:
                break
            if cur_frame in frames:
                yield frame[:, :, ::-1]
                cnt += 1
            cur_frame += 1

    def get_all_frames(self, filename):
        video_capture = cv2.VideoCapture()
        if not video_capture.open(filename):
          print(sys.stderr, 'Error: Cannot open video file ' + filename)
          return
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print('{} has {} frames, sample {} frames.'.format(filename, frame_count, frame_count))
        progress_bar = tqdm(total=frame_count, unit='frame'.format(filename), miniters=1, desc="extract {}".format(filename))
        while True:
            has_frames, frame = video_capture.read()
            if not has_frames:
                break
            progress_bar.update(1)
            yield frame[:, :, ::-1]
        progress_bar.close()

    #等频率抽取n+1帧; 第一帧及最后一帧放入
    def get_frames_n_split(self, filename, n):
        video_capture = cv2.VideoCapture()
        if not video_capture.open(filename):
            print(sys.stderr, 'Error: Cannot open video file ' + filename)
            return
        frame_all = []
        #得到所有要产生frames的对应频率
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        step = frame_count // n
        frames = set([])
        cur_frame = 0
        frames.add(cur_frame)
        while True:
            cur_frame += step
            if cur_frame >= frame_count:
                break
            frames.add(cur_frame)
        frames.add(frame_count - 1)
        print('{} has {} frames, sample {} frames.'.format(filename, frame_count, len(frames)))
        cur_frame = 0
        while True:
            has_frames, frame = video_capture.read()
            if not has_frames:
                break
            if cur_frame in frames:
                yield frame[:, :, ::-1]
            cur_frame += 1

    def extract_youtube8m_feat(self, feat_dict, test_file, youtube8m_path, save):
        if self.extract_youtube8m:
            start_time = time.time()
            if youtube8m_path is not None and os.path.exists(youtube8m_path):
                print(youtube8m_path + ' exist.')
                feat_dict['youtube8m'] = np.load(youtube8m_path)
            else:
                feat_dict['youtube8m'] = []
                for x in  self.gen_batch(self.get_rgb_list(test_file, 3), self.batch_size):
                    t = random.randint(0,1)
                    feat_dict['youtube8m'].extend(self.youtube8m_extractors[t].extract_rgb_frame_features_list(x, self.batch_size))
            if save:
                np.save(youtube8m_path, feat_dict['youtube8m'])
            end_time = time.time()
            print("{}: youtube8m extract cost {} sec".format(test_file, end_time - start_time))
        return feat_dict

    def extract_vggish_feat(self, feat_dict, test_file, vggish_path, save):
        if self.extract_vggish:
            start_time = time.time()
            if vggish_path is not None and os.path.exists(vggish_path):
                print(vggish_path + ' exist.')
                feat_dict['vggish'] = np.load(vggish_path)
            else:
                output_audio = test_file.replace('.mp4', '.wav')
                self.trans2audio(test_file, output_audio)
                if os.path.exists(output_audio):
                    examples_batch = vggish_input.wavfile_to_examples(output_audio)
                    [embedding_batch] = self.audio_sess.run([self.embedding_tensor],
                                                            feed_dict={self.features_tensor: examples_batch})
                    feat_dict['vggish'] = self.pproc.postprocess(embedding_batch)
                    if save:
                        np.save(vggish_path, feat_dict['vggish'])
                else:
                    feat_dict['vggish'] = []
            end_time = time.time()
            print("{}: vggish extract cost {} sec".format(test_file, end_time - start_time))
        return feat_dict

    def extract_ocr_feat(self, feat_dict, test_file, ocr_path, save):
        if self.extract_ocr:
            start_time = time.time()
            if ocr_path is not None and os.path.exists(ocr_path):
                print(ocr_path + ' exist.')
                with open(ocr_path, 'r') as f:
                    feat_dict['ocr'] = f.readline().strip('\n').split('\x001')
            else:
                feat_dict['ocr'] = []
                for x in self.gen_batch(self.get_rgb_list(test_file, 3), self.batch_size):
                    feat_dict['ocr'].extend(self.ocr_extractor.request(x))
                if save:
                    with open(ocr_path, 'w') as f:
                        f.write('\x001'.join(feat_dict['ocr']))
            end_time = time.time()
            print("{}: ocr extract cost {} sec".format(test_file, end_time - start_time))
        return feat_dict

    def extract_asr_feat(self, feat_dict, test_file, asr_path, save):
        if self.extract_asr:
            start_time = time.time()
            if asr_path is not None and os.path.exists(asr_path):
                print(asr_path + ' exist.')
                with open(asr_path, 'r') as f:
                    feat_dict['asr'] = f.readline().strip('\n').split('\x001')
            else:
                output_audio = test_file.replace('.mp4', '.wav')
                video_asr = ''
                self.trans2audio(test_file, output_audio)
                if os.path.exists(output_audio):
                    try:
                        video_asr = self.asr_extractor.request(output_audio)
                    except:
                        print(output_audio)
                        print(traceback.format_exc())
                feat_dict['asr'] = video_asr
                if save:
                    with open(asr_path, 'w') as f:
                        f.write('\x001'.join(feat_dict['text']))
            end_time = time.time()
            print("{}: asr extract cost {} sec".format(test_file, end_time - start_time))
        return feat_dict

    def extract_feat(self, test_file, youtube8m_path=None, vggish_path=None, ocr_path=None, asr_path=None, save=True):
        feat_dict={}
        feat_dict = self.extract_youtube8m_feat(feat_dict, test_file, youtube8m_path, save)
        feat_dict = self.extract_vggish_feat(feat_dict, test_file, vggish_path, save)
        feat_dict = self.extract_ocr_feat(feat_dict, test_file, ocr_path, save)
        feat_dict = self.extract_asr_feat(feat_dict, test_file, asr_path, save)
        return feat_dict

    def trans2audio(self, test_file, output_audio):
        if not os.path.exists(output_audio):
            command = 'ffmpeg -loglevel error -i '+ test_file + ' ' + output_audio
            os.system(command)
            print("audio file not exists: {}".format(output_audio))

    def get_rgb_list(self, path, mode, n = None, every_ms = None, max_num_frames = None):
        if mode == 1:
            return self.get_frames_n_split(path, n)
        elif mode == 2:
            return self.get_frames_same_interval(path, every_ms, max_num_frames)
        else:
            return self.get_all_frames(path)

    def gen_batch(self, generator, batch_size):
        batch = []
        count = 0
        for x in generator:
            batch.append(x)
            count += 1
            if count == batch_size:
                yield batch
                batch = []
                count = 0
        if count > 0:
            yield batch
