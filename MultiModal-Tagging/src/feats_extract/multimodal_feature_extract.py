from __future__ import unicode_literals
import sys,os
import numpy as np
import cv2
import time
import tensorflow as tf
import json
import traceback

from src.feats_extract.imgfeat_extractor.youtube8M_extractor import YouTube8MFeatureExtractor
from src.feats_extract.imgfeat_extractor.finetuned_resnet101 import FinetunedResnet101Extractor
from src.feats_extract.txt_extractor.text_requests import VideoASR,VideoOCR,ImageOCR
from src.feats_extract.audio_extractor import vggish_input,vggish_params,vggish_postprocess,vggish_slim
from src.dataloader.preprocess.cnn_preprocessing import inception_preprocessing
import utils.utils as utils

BASE = "/home/tione/notebook/VideoStructuring/MultiModal-Tagging/"
PCA_PARAMS_PATH = BASE + "pretrained/vggfish/vggish_pca_params.npz"
VGGISH_CHECKPOINT_PATH = BASE + "pretrained/vggfish/vggish_model.ckpt"
VIDEO_EXTRACTOR = 'Youtube8M'
MODE = 1    #1表示将视频等分成n份, 2表示将取视频前n帧，每帧间隔 x ms
EVERY_MS = 1000
MAX_NUM_FRAMES = 300
N = 100

class MultiModalFeatureExtract(object):

    def get_video_extractor(self, batch_size):
        if VIDEO_EXTRACTOR == 'Youtube8M':
            return YouTube8MFeatureExtractor(use_batch=batch_size != 1)
        elif VIDEO_EXTRACTOR == 'FinetunedResnet101':
            return FinetunedResnet101Extractor()
        else:
            raise NotImplementedError(VIDEO_EXTRACTOR)

    """docstring for ClassName"""
    def __init__(self, batch_size = 1,
                 extract_video = True,
                 extract_img = True,
                 extract_audio = True,
                 extract_ocr = True,
                 extract_asr = True,
                 use_gpu = True,
                 ):
        super(MultiModalFeatureExtract, self).__init__()
        self.extract_video = extract_video
        self.extract_img = extract_img
        self.extract_audio = extract_audio
        self.extract_ocr = extract_ocr
        self.extract_asr = extract_asr
        self.batch_size = batch_size

        #视频特征抽取模型
        if extract_video:
            self.video_extractor = self.get_video_extractor(batch_size)

        #音频特征抽取模型
        if extract_audio:
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

        self.error_log = open('/home/tione/notebook/VideoStructuring/err.log', 'w')

    def close():
        self.error_log.close()

    def get_frames_same_interval(self, filename, every_ms=1000, max_num_frames=300):
        video_capture = cv2.VideoCapture()
        if not video_capture.open(filename):
          print(sys.stderr, 'Error: Cannot open video file ' + filename)
          return
        last_ts = -99999  # The timestamp of last retrieved frame.
        num_retrieved = 0

        frame_all = []
        while num_retrieved < max_num_frames:
            # Skip frames
            while video_capture.get(cv2.CAP_PROP_POS_MSEC) < every_ms + last_ts:
                if not video_capture.read()[0]:
                    return frame_all

            last_ts = video_capture.get(cv2.CAP_PROP_POS_MSEC)
            has_frames, frame = video_capture.read()
            if not has_frames:
                break
            frame_all.append(frame[:, :, ::-1])
            num_retrieved += 1
        return frame_all

    def extract_video_feat(self, feat_dict, video_file, frames, video_npy_path, save):
        if self.extract_video:
            start_time = time.time()
            feat_dict['video'] = []
            if video_npy_path is not None and os.path.exists(video_npy_path):
                feat_dict['video'] = np.load('video_npy_path')
            else:
                cap = cv2.VideoCapture(video_file)
                for r_index, r_frame, r_time, count in self.gen_img_batch(cap, youtube8m_dir, frames, self.batch_size, 'extract youtube8m feat {}'.format(video_file)):
                    feat_dict['video'].extend(self.youtube8m_extractor.extract_rgb_frame_features_list(r_frame, count))
                cap.release()
                if save:
                    np.save(video_npy_path, feat_dict['video'])
            end_time = time.time()
            self.error_log.write("{}: youtube8m extract cost {} sec.\n".format(video_file, end_time - start_time))

    def extract_img_feat(self, feat_dict, video_file, frame, image_jpg_path, save):
        if self.extract_img:
            start_time = time.time()
            if image_jpg_path is not None and os.path.exists(image_jpg_path):
                feat_dict['image'] = cv2.imread(image_jpg_path, 1)
            else:
                cap = cv2.VideoCapture(video_file)
                r_index, r_frame, r_time, count = self.gen_img_list(cap, [frames])
                feat_dict['image'] = r_frame[0]
                cap.release()
                if save:
                    cv2.imwrite(image_jpg_path, feat_dict['image'])
            end_time = time.time()
            self.error_log.write("{}: img extract cost {} sec.\n".format(video_file, end_time - start_time))

    def extract_audio_feat(self, feat_dict, audio_file, audio_npy_path, save):
        if self.extract_audio:
            start_time = time.time()
            if audio_npy_path is not None and os.path.exists(audio_npy_path):
                feat_dict['audio'] = np.load(audio_npy_path)
            else:
                if os.path.exists(audio_file):
                    examples_batch = vggish_input.wavfile_to_examples(audio_file)
                    [embedding_batch] = self.audio_sess.run([self.embedding_tensor],
                                                            feed_dict={self.features_tensor: examples_batch})
                    feat_dict['audio'] = self.pproc.postprocess(embedding_batch)
                    if save:
                        np.save(audio_npy_path, feat_dict['audio'])
                else:
                    feat_dict['audio'] = []
            end_time = time.time()
            self.error_log.write("{}: audio extract cost {} sec.\n".format(audio_file, end_time - start_time))
        return feat_dict

    def extract_ocr_feat(self, video_file, frames, ocr_path, save):
        if self.extract_ocr:
            start_time = time.time()
            if ocr_path is not None and os.path.exists(ocr_path):
                with open(ocr_path, 'r') as f:
                    feat_dict['ocr'] = f.readline().strip('\n').split('|')
            else:
                cap = cv2.VideoCapture(video_file)
                if len(frames) > 3:
                    s = frames[0]
                    e = frames[1]
                    m = frames[len(frames) // 2]
                    frames = [s, m, e]
                r_index, r_frame, r_time, count = self.gen_img_list(cap, frames)
                for frame in r_frame:
                    feat_dict['ocr'].extend(self.ocr_extractor.request(frame))
                if save:
                    with open(ocr_path, 'w') as f:
                        f.write('|'.join(feat_dict['ocr']))
                cap.release()

            end_time = time.time()
            self.error_log.write("{}: ocr extract cost {} sec.\n".format(video_file, end_time - start_time))

    def extract_asr_feat(self, feat_dict, audio_file, asr_file_path, save):
        if self.extract_asr:
            start_time = time.time()
            video_asr = ''
            if asr_file_path is not None and os.path.exists(asr_file_path):
                with open(asr_file_path, 'r') as f:
                    feat_dict['asr'] = f.readline().strip('\n')
            else:
                if audio_file is not None and os.path.exists(audio_file):
                    try:
                        video_asr = self.asr_extractor.request(audio_file)
                    except:
                        print(audio_file)
                        print(traceback.format_exc())
                    feat_dict['asr'] = video_asr
                    if save:
                        with open(asr_file_path, 'w') as f:
                            f.write(feat_dict['asr'])
            end_time = time.time()
            self.error_log.("{}: asr extract cost {} sec.\n".format(video_file, end_time - start_time))
        return feat_dict

    def extract_feat(self, video_file, video_npy_path=None, text_txt_path=None, audio_npy_path=None, img_jpg_path=None, ocr_file_path=None, asr_file_path=None, save=True):
        cap = cv2.VideoCapture(video_file)
        frame_count, fps, h, w, ts = self.read_video_info(cap)
        cap.release()

        feat_dict={}
        #frames = utils.get_frames_same_interval(frame_count, sample_fps)
        frames = utils.get_frames_same_ts_interval(ts, 1.0, 300)

        audio_file = video_file.replace('.mp4', '.wav')
        self.trans2audio(video_file, audio_file)

        feat_dict = self.extract_video_feat(feat_dict, video_file, frames, video_npy_path, save)
        feat_dict = self.extract_img_feat(feat_dict, video_file, sorted(list(frames))[len(frames) // 2], video_npy_path, save)
        feat_dict = self.extract_audio_feat(feat_dict, audio_file, audio_npy_path, save)
        feat_dict = self.extract_ocr_feat(feat_dict, video_file, frames, ocr_file_path, save)
        feat_dict = self.extract_asr_feat(feat_dict, video_file, asr_file_path, save)
        feat_dict['text'] = {'video_ocr': feat_dict['ocr'], 'video_asr': feat_dict['asr']}
        with open(text_txt_path, 'w') as f:
            f.write(json.dumps(feat_dict['text'], ensure_ascii=False))
        return feat_dict

    def gen_img_list(self, cap, frames):
        r_frame = []
        r_index = []
        r_time = []
        count = 0
        index = 0
        while True:
            has_frame, frame = cap.read()
            if not has_frame:
                break
            ts = cap.get(cv2.CAP_PROP_POS_MSEC)
            if index in frames:
                r_index.append(index)
                r_frame.append(frame)
                r_time.append(ts)
                count += 1
            index += 1
        return r_frame, r_index, r_time, count

    def gen_img_batch(self, cap, frames, batch_size, desc):
        r_frame = []
        r_index = []
        r_time = []
        count = 0
        index = 0
        progress_bar = tqdm(total=len(frames), miniters=1, desc=desc)
        while True:
            has_frame, frame = cap.read()
            if not has_frame:
                break
            ts = cap.get(cv2.CAP_PROP_POS_MSEC)
            if index in frames:
                r_index.append(index)
                r_frame.append(frame)
                r_time.append(ts)
                progress_bar.update(1)
                count += 1
            if count == batch_size:
                yield r_index, r_frame, r_time, count
                count = 0
                r_index = []
                r_frame = []
                r_time = []
            index += 1
        if count > 0:
            yield r_index, r_frame, r_time, count
        progress_bar.close()

    def trans2audio(self, video_file, output_audio):
        if not os.path.exists(output_audio):
            command = 'ffmpeg -loglevel error -i '+ video_file + ' ' + output_audio
            os.system(command)
            print("audio file not exists: {}".format(output_audio))

    def read_video_info(self, cap):
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(cv2.CAP_PROP_FPS)
        ts = []
        while True:
            has_frame, frame = cap.read()
            if not has_frame:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC)
            if t == 0 and len(ts) > 0:
                continue
            ts.append(t)
        return frame_count, fps, h, w, ts
