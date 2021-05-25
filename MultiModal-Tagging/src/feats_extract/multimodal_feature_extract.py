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

FRAMES_PER_SECOND = 1
PCA_PARAMS = "pretrained/vggfish/vggish_pca_params.npz" #'Path to the VGGish PCA parameters file.'
VGGISH_CHECKPOINT = 'pretrained/vggfish/vggish_model.ckpt'
CAP_PROP_POS_MSEC = 0

class MultiModalFeatureExtract(object):
    """docstring for ClassName"""
    def __init__(self, batch_size = 1, 
                 imgfeat_extractor = 'Youtube8M',
                 extract_video = True,
                 extract_audio = True,
                 extract_text = True,
                 extract_img = True,
                 use_gpu = True):
        super(MultiModalFeatureExtract, self).__init__()
        self.extract_video = extract_video
        self.extract_audio = extract_audio
        self.extract_text = extract_text
        self.extract_img = extract_img
        
        if extract_video:   #视频特征抽取模型
            self.batch_size = batch_size
            if imgfeat_extractor == 'Youtube8M':
                self.extractor = YouTube8MFeatureExtractor(use_batch = batch_size!=1)
            elif imgfeat_extractor == 'FinetunedResnet101':
                self.extractor = FinetunedResnet101Extractor()
            else:
                raise NotImplementedError(imgfeat_extractor)
                
        if extract_audio:   #音频特征抽取模型
            self.pproc = vggish_postprocess.Postprocessor(PCA_PARAMS)  # audio pca
            self.audio_graph = tf.Graph()
            config = tf.ConfigProto(allow_soft_placement=True,
                                 log_device_placement=True)
            config.gpu_options.allow_growth = True
            with self.audio_graph.as_default():
                # 音频
                self.audio_sess = tf.Session(graph=self.audio_graph, config=config)
                vggish_slim.define_vggish_slim(training=False)
                vggish_slim.load_vggish_slim_checkpoint(self.audio_sess, VGGISH_CHECKPOINT)
            self.features_tensor = self.audio_sess.graph.get_tensor_by_name(
                vggish_params.INPUT_TENSOR_NAME)
            self.embedding_tensor = self.audio_sess.graph.get_tensor_by_name(
                vggish_params.OUTPUT_TENSOR_NAME)
            
        if extract_text:    #文本特征抽取模型
            self.video_ocr_extractor = VideoOCR(use_gpu)
            self.video_asr_extractor = VideoASR(use_gpu)
            self.image_ocr_extractor = ImageOCR(use_gpu)

    def frame_iterator(self, filename, every_ms=1000, max_num_frames=300):
        """Uses OpenCV to iterate over all frames of filename at a given frequency.

        Args:
          filename: Path to video file (e.g. mp4)
          every_ms: The duration (in milliseconds) to skip between frames.
          max_num_frames: Maximum number of frames to process, taken from the
            beginning of the video.

        Yields:
          RGB frame with shape (image height, image width, channels)
        """
        video_capture = cv2.VideoCapture()
        if not video_capture.open(filename):
          print(sys.stderr, 'Error: Cannot open video file ' + filename)
          return
        last_ts = -99999  # The timestamp of last retrieved frame.
        num_retrieved = 0

        while num_retrieved < max_num_frames:
          # Skip frames
          while video_capture.get(CAP_PROP_POS_MSEC) < every_ms + last_ts:
            if not video_capture.read()[0]:
              return

          last_ts = video_capture.get(CAP_PROP_POS_MSEC)
          has_frames, frame = video_capture.read()
          if not has_frames:
            break
          yield frame
          num_retrieved += 1
    def frame_iterator_list(self, filename, every_ms=1000, max_num_frames=300):
        video_capture = cv2.VideoCapture()
        if not video_capture.open(filename):
          print(sys.stderr, 'Error: Cannot open video file ' + filename)
          return
        last_ts = -99999  # The timestamp of last retrieved frame.
        num_retrieved = 0

        frame_all = []
        while num_retrieved < max_num_frames:
            # Skip frames
            while video_capture.get(CAP_PROP_POS_MSEC) < every_ms + last_ts:
                if not video_capture.read()[0]:
                    return frame_all

            last_ts = video_capture.get(CAP_PROP_POS_MSEC)
            has_frames, frame = video_capture.read()
            if not has_frames:
                break
            frame_all.append(frame[:, :, ::-1])
            num_retrieved += 1

        return frame_all

    def extract_feat(self, test_file,
                     frame_npy_path=None, audio_npy_path=None, txt_file_path=None,
                     image_jpg_path=None, save=True):
        
        #根据文件类型选择特征抽取方式
        filetype = test_file.split('.')[-1]
        if filetype in ['mp4', 'avi']:
            feat_dict = self.extract_video_feat(test_file, frame_npy_path, audio_npy_path, txt_file_path, image_jpg_path, save)
        elif filetype in ['jpg', 'png']:
            feat_dict = self.extract_image_feat(test_file)
        else:
            raise NotImplementedError
        return feat_dict

    def extract_image_feat(self, test_file):
        feat_dict={}
        feat_dict['image'] = cv2.imread(test_file,1)[:,:,::-1] #convert to rgb

        if self.extract_text:
            start_time = time.time()
            image_ocr = self.image_ocr_extractor.request(test_file)        
            feat_dict['text'] = json.dumps({'image_ocr': image_ocr}, ensure_ascii=False)
            end_time = time.time()
            print("text extract cost {} sec".format(end_time - start_time))   
        return feat_dict
    
    def extract_video_feat(self, test_file,
                     frame_npy_path=None, audio_npy_path=None, txt_file_path=None,
                     image_jpg_path=None, save=True):
        feat_dict={}
        #=============================================视频
        start_time = time.time()
        if frame_npy_path is not None and os.path.exists(frame_npy_path) and self.extract_video:
            feat_dict['video'] = np.load(frame_npy_path)
        elif self.extract_video:
            if self.batch_size == 1:
                features_arr = []
                for rgb in self.frame_iterator(test_file, every_ms=1000.0/FRAMES_PER_SECOND):
                    features = self.extractor.extract_rgb_frame_features(rgb[:, :, ::-1])
                    features_arr.append(features)
                feat_dict['video'] = features_arr
            else:
                rgb_list = self.frame_iterator_list(test_file, every_ms=1000.0/FRAMES_PER_SECOND)
                feat_dict['video'] = self.extractor.extract_rgb_frame_features_list(rgb_list, self.batch_size)
            if save:
                np.save(frame_npy_path, feat_dict['video'])
        end_time = time.time()
        print("video extract cost {} sec".format(end_time - start_time))   
        #=============================================图片抽帧
        rgb_list = None
        start_time = time.time()
        if image_jpg_path is not None and os.path.exists(image_jpg_path) and self.extract_img:
            feat_dict['image'] = cv2.imread(image_jpg_path)[:, :, ::-1]
        elif self.extract_img:
            rgb_list = self.frame_iterator_list(test_file, every_ms=1000.0/FRAMES_PER_SECOND)
            feat_dict['image'] = rgb_list[len(rgb_list)//2]
            if save:
                cv2.imwrite(image_jpg_path, feat_dict['image'][:,:,::-1])
        end_time = time.time()
        print("image extract cost {} sec".format(end_time - start_time))
        #=============================================音频
        start_time = time.time()
        if audio_npy_path is not None and os.path.exists(audio_npy_path) and self.extract_audio:
            feat_dict['audio'] = np.load(audio_npy_path)
        elif self.extract_audio:
            output_audio = test_file.replace('.mp4','.wav')
            self.trans2audio(test_file, output_audio)
            if os.path.exists(output_audio):
                examples_batch = vggish_input.wavfile_to_examples(output_audio)
                [embedding_batch] = self.audio_sess.run([self.embedding_tensor],
                        feed_dict={self.features_tensor: examples_batch})
                feat_dict['audio'] = self.pproc.postprocess(embedding_batch)
                if save:
                    np.save(audio_npy_path, feat_dict['audio'])
            else:
                feat_dict['audio'] = []
        end_time = time.time()
        print("audio extract cost {} sec".format(end_time - start_time)) 
        #=============================================文本
        start_time = time.time()
        if txt_file_path is None and os.path.exists(txt_file_path) and self.extract_text:
            print(txt_file_path + ' exist.')
            with open(txt_file_path, 'r') as f:
                feat_dict['text'] = f.readline().strip('\n')
        elif self.extract_text:
            video_ocr = ''
            if rgb_list != None:
                video_ocr = self.video_ocr_extractor.request(rgb_list)
            else:
                video_ocr = self.video_ocr_extractor.request(self.frame_iterator_list(test_file, every_ms=1000.0/FRAMES_PER_SECOND))
            output_audio = test_file.replace('.mp4','.wav')
            video_asr = ''
            self.trans2audio(test_file, output_audio)
            if os.path.exists(output_audio):
                try:
                    video_asr = self.video_asr_extractor.request(output_audio)
                except:
                    print(output_audio)
                    print(traceback.format_exc())
            feat_dict['text'] = json.dumps({'video_ocr': video_ocr, 'video_asr': video_asr}, ensure_ascii=False)
            if save:
                with open(txt_file_path, 'w') as f:
                    f.write(feat_dict['text'])
        end_time = time.time()
        print("text extract cost {} sec".format(end_time - start_time)) 
        return feat_dict

    def trans2audio(self, test_file, output_audio):
        if not os.path.exists(output_audio):
            command = 'ffmpeg -loglevel error -i '+ test_file + ' ' + output_audio
            os.system(command)
            print("audio file not exists: {}".format(output_audio))

