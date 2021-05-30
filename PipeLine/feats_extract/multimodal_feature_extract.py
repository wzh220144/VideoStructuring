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
from pydub import AudioSegment

BASE = "/home/tione/notebook/VideoStructuring"
PCA_PARAMS_PATH = BASE + "/pretrained/vggfish/vggish_pca_params.npz"
VGGISH_CHECKPOINT_PATH = BASE + "/pretrained/vggfish/vggish_model.ckpt"
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

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
        self.error_log = open('/home/tione/notebook/VideoStructuring/PipeLine/err.log', 'w')
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

    def get_frames_same_interval(self, frame_count, interval):
        frames = set([])
        cur_frame = 0
        while cur_frame < frame_count:
            frames.add(cur_frame)
            cur_frame += interval
        if frame_count > 0:
            frames.add(frame_count - 1)
        return frames

    def get_frames_n_split(self, frame_count, n):
        step = min(frame_count // n, 1)
        frames = set([])
        cur_frame = 0
        while cur_frame < frame_count:
            frames.add(cur_frame)
            cur_frame += step
        if frame_count > 0:
            frames.add(frame_count - 1)
        return frames

    def get_all_frames(self, frame_count):
        return range(frame_count)

    def extract_youtube8m_feat(self, video_file, frames, youtube8m_dir, save):
        if self.extract_youtube8m:
            start_time = time.time()
            cap = cv2.VideoCapture(video_file)
            for r_index, r_frame, r_time, count in self.gen_img_batch(cap, youtube8m_dir, frames, self.batch_size, 'extract youtube8m feat {}'.format(video_file)):
                feats = self.youtube8m_extractor.extract_rgb_frame_features_list(r_frame, count)
                if save:
                    for i in range(count):
                        t = youtube8m_dir + '#' + str(r_index[i]) + '.npy'
                        np.save(t, feats[i])
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
            progress_bar = tqdm(total=len(split_audio_files), miniters=1, desc='extract stft feat {}'.format(video_file))
            for audio_file in split_audio_files:
                progress_bar.update(1)
                feat_path = stft_dir + audio_file.split("/")[-1].split('.')[0] + '.npy'
                if os.path.exists(feat_path):
                    self.error_log.write('{} exist.\n'.format(feat_path))
                    continue
                try:
                    t = self.stft_extractor.extract_stft(audio_file)
                    if save:
                        np.save(feat_path, t)
                except Exception as e:
                    self.error_log.write("extract vggish from {} failed: {}\n".format(audio_file, e))
            end_time = time.time()
            self.error_log.write("{}: stft extract cost {} sec.\n".format(video_file, end_time - start_time))
            progress_bar.close()

    def extrat_resnet50_feat(self, video_file, frames, resnet50_dir, save):
        if self.extract_resnet50:
            start_time = time.time()
            end_time = time.time()
            print("{}: resnet50 extract cost {} sec".format(video_file, end_time - start_time))

    def read_video_info(self, cap):
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(cv2.CAP_PROP_FPS)
        return frame_count, fps, h, w

    def extract_feat(self, video_file, split_dir, sample_fps,
                     youtube8m_dir=None, resnet50_dir=None,
                     vggish_dir=None, stft_dir=None,
                     ocr_dir=None, asr_dir=None,
                     save=True):
        cap = cv2.VideoCapture(video_file)
        frame_count, fps, h, w = self.read_video_info(cap)
        cap.release()

        frames = self.get_frames_same_interval(frame_count, sample_fps)
        #print('{} has {} frames, sample {} frames.'.format(video_file, frame_count, len(frames)))

        audio_file = video_file.replace('.mp4', '.wav')
        self.trans2audio(video_file, audio_file)
        split_audio_files = self.split_audio(video_file, audio_file, frames, split_dir)

        self.extract_youtube8m_feat(video_file, frames, youtube8m_dir, save)
        self.extrat_resnet50_feat(video_file, frames, resnet50_dir, save)
        self.extract_vggish_feat(video_file, split_audio_files, vggish_dir, save)
        self.extract_stft_feat(video_file, split_audio_files, stft_dir, save)
        self.extract_ocr_feat(video_file, frames, ocr_dir, save)
        self.extract_asr_feat(video_file, split_audio_files, asr_dir, save)

    def split_audio(self, video_file, audio_file, frames, split_audio_dir):
        video_id = audio_file.split('/')[-1].split('.')[0]
        cap = cv2.VideoCapture(video_file)
        seg = AudioSegment.from_wav(audio_file)
        max_len = len(seg)
        res = []
        for pre_index, cur_index, pre, cur in self.gen_ts_interval(cap, frames, 'split audio {}:'.format(audio_file)):
            split_audio_file = '{}/{}#{}#{}.wav'.format(split_audio_dir, video_id, pre_index, cur_index)
            res.append(split_audio_file)
            if not os.path.exists(split_audio_file):
                seg[int(pre): min(int(cur), max_len)].export(split_audio_file, format='wav')
                #print("audio file not exists: {}".format(split_audio_file))
        return res

    def trans2audio(self, video_file, audio_file):
        if not os.path.exists(audio_file):
            command = 'ffmpeg -loglevel error -i '+ video_file + ' ' + audio_file
            os.system(command)

    def gen_img_batch(self, cap, feat_dir, frames, batch_size, desc):
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
                t = feat_dir + str(index) + '.npy'
                if not os.path.exists(t):
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

    def gen_ts_interval(self, cap, frames, desc):
        count = 0
        #progress_bar = tqdm(total=len(frames), miniters=1, desc=desc)
        has_frame, frame = cap.read()
        if not has_frame:
            return
        pre = 0
        pre_index = 0
        cur_index = 0
        while True:
            has_frame, frame = cap.read()
            if not has_frame:
                break
            cur_index += 1
            cur = cap.get(cv2.CAP_PROP_POS_MSEC)
            if cur_index in frames:
                yield pre_index, cur_index, pre, cur
                count += 1
                pre = cur
                pre_index = cur_index
                #progress_bar.update(count)
        #progress_bar.update(count)
        #progress_bar.close()
