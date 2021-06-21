from __future__ import unicode_literals
import os
import numpy as np
import cv2
import time
import tensorflow as tf
from tqdm import tqdm

from feats_extract.imgfeat_extractor.youtube8M_extractor import YouTube8MFeatureExtractor
from feats_extract.audio_extractor.stft_extractor import StftExtractor
from feats_extract.txt_extractor.text_requests import VideoASR,VideoOCR
from feats_extract.audio_extractor import vggish_input,vggish_params,vggish_postprocess,vggish_slim
from utils import utils
from pydub import AudioSegment

BASE = "/home/tione/notebook/VideoStructuring"
PCA_PARAMS_PATH = BASE + "/pretrained/vggfish/vggish_pca_params.npz"
VGGISH_CHECKPOINT_PATH = BASE + "/pretrained/vggfish/vggish_model.ckpt"

class MultiModalFeatureExtract(object):

    """docstring for ClassName"""
    def __init__(self, batch_size = 1,
                 extract_youtube8m = True,
                 extract_resnet50 = True,
                 extract_vggish = True,
                 extract_stft = True,
                 extract_ocr = True,
                 extract_asr = True,
                 use_gpu = True,
                 device = 'cuda',
                 ):
        super(MultiModalFeatureExtract, self).__init__()
        self.error_log = open('/home/tione/notebook/VideoStructuring/PipeLine/feat_err.log', 'w')
        self.extract_youtube8m = extract_youtube8m
        self.extract_resnet50 = extract_resnet50
        self.extract_vggish = extract_vggish
        self.extract_stft = extract_stft
        self.extract_ocr = extract_ocr
        self.extract_asr = extract_asr
        self.batch_size = batch_size

        if extract_youtube8m:
            self.youtube8m_extractor = YouTube8MFeatureExtractor(device, use_batch = batch_size != 1)

        if extract_resnet50:
            pass

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

        if extract_stft:
            self.stft_extractor = StftExtractor()

        if extract_ocr:
            self.ocr_extractor = VideoOCR(use_gpu)

        if extract_asr:
            self.asr_extractor = VideoASR(use_gpu)

    def extract_youtube8m_feat(self, video_file, frames, youtube8m_dir, save):
        if self.extract_youtube8m:
            start_time = time.time()
            cap = cv2.VideoCapture(video_file)
            fps = cap.get(cv2.CAP_PROP_FPS)
            video_id = video_file.split('/')[-1].split('.')[0]
            os.makedirs(youtube8m_dir, exist_ok=True)
            for r_start_frame, r_end_frame, r_index, r_frame, cnt in self.gen_img_batch(cap, video_file, youtube8m_dir, fps, frames, self.batch_size):
                feats = self.youtube8m_extractor.extract_rgb_frame_features_list(r_frame, cnt)
                if save:
                    for i in range(cnt):
                        start_frame = r_start_frame[i]
                        end_frame = r_end_frame[i]
                        index = r_index[i]
                        feat_file = '{}/{}#{}#{}#{}.npy'.format(youtube8m_dir, index, start_frame, end_frame, int(fps))
                        np.save(feat_file, feats[i])
            end_time = time.time()
            cap.release()
            self.error_log.write("{}: youtube8m extract cost {} sec.\n".format(video_file, end_time - start_time))

    def _extract_vggish_feat(self, batch, feat_paths, save):
        res = self.audio_sess.run([self.embedding_tensor], feed_dict={self.features_tensor: batch})
        for i in range(len(batch)):
            if save:
                np.save(feat_paths[i], res[i])

    def extract_vggish_feat(self, video_file, split_audio_files, vggish_dir, save):
        if self.extract_vggish:
            start_time = time.time()
            batch = []
            count = 0
            feat_paths = []
            for audio_file in split_audio_files:
                feat_path = vggish_dir + audio_file.split("/")[-1].split('.')[0] + '.npy'
                if os.path.exists(feat_path):
                    self.error_log.write('{} exist.\n'.format(feat_path))
                    continue
                try:
                    batch.append(vggish_input.wavfile_to_examples(audio_file))
                    count += 1
                    feat_paths.append(feat_path)
                except Exception as e:
                    self.error_log.write("extract vggish from {} failed: {}\n".format(audio_file, e))
                if count == self.batch_size:
                    self._extract_vggish_feat(batch, feat_paths, save)
                    feat_paths = []
                    batch = []
                    count = 0
            if count > 0:
                self._extract_vggish_feat(batch, feat_paths, save)
            end_time = time.time()
            self.error_log.write("{}: vggish extract cost {} sec.\n".format(video_file, end_time - start_time))

    def extract_ocr_feat(self, video_file, frames, ocr_dir, save):
        if self.extract_ocr:
            start_time = time.time()
            cap = cv2.VideoCapture(video_file)
            for r_index, r_frame, r_time, count in self.gen_img_batch(cap, ocr_dir, frames, self.batch_size, 'extract ocr feat {}:'.format(video_file)):
                feats = self.ocr_extractor.request(r_frame, count)
                if save:
                    for i in range(count):
                        feat_path = ocr_dir + '#' + str(r_index[i]) + '.txt'
                        t = feats[i]
                        with open(feat_path, 'w') as fd:
                            fd.write(t)
            end_time = time.time()
            cap.release()
            self.error_log.write("{}: ocr extract cost {} sec.\n".format(video_file, end_time - start_time))

    def extract_asr_feat(self, video_file, split_audio_files, asr_dir, save):
        if self.extract_asr:
            start_time = time.time()
            for audio_file in split_audio_files:
                feat_path = asr_dir + audio_file.split("/")[-1].split('.')[0] + '.npy'
                if os.path.exists(feat_path):
                    self.error_log.write('{} exist.\n'.format(feat_path))
                    continue
                t = self.asr_extractor.request(audio_file)
                if save:
                    with open(feat_path, 'w') as fd:
                        fd.write(t)
            end_time = time.time()
            self.error_log.write("{}: asr extract cost {} sec\n".format(video_file, end_time - start_time))

    def extract_stft_feat(self, video_file, split_audio_files, stft_dir, save):
        if self.extract_stft:
            start_time = time.time()
            for audio_file in split_audio_files:
                index = audio_file.split('/')[-1].split('.')[0]
                vid = audio_file.split('/')[-2]
                os.makedirs(stft_dir, exist_ok=True)
                feat_path = '{}/{}.npy'.format(stft_dir, index)
                if os.path.exists(feat_path):
                    self.error_log.write('{} exist.\n'.format(feat_path))
                    continue
                try:
                    self.error_log.write('{} do not exist.\n'.format(feat_path))
                    t = self.stft_extractor.extract_stft(audio_file)
                    if save:
                        np.save(feat_path, t)
                except Exception as e:
                    self.error_log.write("extract stft from {} failed: {}\n".format(audio_file, e))
            end_time = time.time()
            self.error_log.write("{}: stft extract cost {} sec.\n".format(video_file, end_time - start_time))

    def extrat_resnet50_feat(self, video_file, frames, resnet50_dir, save):
        if self.extract_resnet50:
            start_time = time.time()
            end_time = time.time()
            print("{}: resnet50 extract cost {} sec".format(video_file, end_time - start_time))

    def extract_feat(self, video_file, shot_dir, sample_fps,
                     youtube8m_dir=None, resnet50_dir=None,
                     vggish_dir=None, stft_dir=None,
                     ocr_dir=None, asr_dir=None,
                     save=True):
        cap = cv2.VideoCapture(video_file)
        frame_count, fps, h, w = utils.read_video_info(cap)
        cap.release()
    
        frames, split_video_files, split_audio_files, flag = utils.read_shot_info(video_file, shot_dir)

        if flag:
            self.error_log.write('start extract feat of {}\n'.format(video_file))
            self.extract_youtube8m_feat(video_file, frames, youtube8m_dir, save)
            self.extrat_resnet50_feat(video_file, frames, resnet50_dir, save)
            self.extract_vggish_feat(video_file, split_audio_files, vggish_dir, save)
            self.extract_stft_feat(video_file, split_audio_files, stft_dir, save)
            self.extract_ocr_feat(video_file, frames, ocr_dir, save)
            self.extract_asr_feat(video_file, frames, asr_dir, save)

    def gen_img_batch(self, cap, video_file, feat_dir, fps, frames, batch_size):
        video_id = video_file.split('/')[-1].split('.')[0]
        has_frame, frame = cap.read()
        if not has_frame:
            return
        r_start_frame = []
        r_end_frame = []
        r_index = []
        r_frame = []
        cur_frames = [frame]
        start_frame = 0
        index = 0
        cnt = 0
        frame_index_dict = {}
        sorted_frames = sorted(list(frames))
        for i in range(1, len(sorted_frames)):
            frame_index_dict[sorted_frames[i]] = i - 1
        while True:
            has_frame, frame = cap.read()
            if not has_frame:
                break
            index += 1
            if index in frames:
                t_index = rame_index_dict[index]
                feat_file = '{}/{}#{}#{}.npy'.format(feat_dir, t_index, start_frame, index, int(fps))
                if not os.path.exists(feat_file):
                    r_start_frame.append(start_frame)
                    r_end_frame.append(index)
                    r_index.append(t_index)
                    r_frame.append(cur_frames[len(cur_frames) // 2])
                    cnt += 1
                    self.error_log.write('{} do not exist.\n'.format(feat_file))
                else:
                    self.error_log.write('{} exist.\n'.format(feat_file))
                cur_frames = [frame]
                start_frame = index
            if cnt % batch_size == 0 and cnt > 0:
                yield r_start_frame, r_end_frame, r_index, r_frame, cnt
                r_start_frame = []
                r_end_frame = []
                r_index = []
                r_frame = []
                cnt = 0
            cur_frames.append(frame)

        if cnt > 0:
            yield r_start_frame, r_end_frame, r_index, r_frame, cnt

    
