# -*- coding: utf-8 -*-
import easyocr
import cv2
import speech_recognition as sr
from pocketsphinx import pocketsphinx, Jsgf, FsgModel
from queue import Queue
import os

'''
class VideoASR():
    def __init__(self, use_gpu, maxsize):


    """视频ASR"""
    def request(self, path):
        with sr.AudioFile(path) as source:
            audio = self.r.record(source)
        return self.recognize_sphinx(audio)
'''
 
class VideoASR():
    def __init__(self, use_gpu, maxsize):
        self.r = sr.Recognizer()
        self.decoder_queue = Queue(maxsize=maxsize)
        for i in range(maxsize):
            print(i)
            self.decoder_queue.put(self.get_decoder())

    def get_decoder(self, language='zh-CN'):
        language_directory = os.path.join("/home/tione/notebook/envs/tf2/lib/python3.6/site-packages/speech_recognition/pocketsphinx-data", language)
        acoustic_parameters_directory = os.path.join(language_directory, "acoustic-model")
        language_model_file = os.path.join(language_directory, "language-model.lm.bin")
        phoneme_dictionary_file = os.path.join(language_directory, "pronounciation-dictionary.dict")
        # create decoder object
        config = pocketsphinx.Decoder.default_config()
        config.set_string("-hmm", acoustic_parameters_directory)  # set the path of the hidden Markov model (HMM) parameter files
        config.set_string("-lm", language_model_file)
        config.set_string("-dict", phoneme_dictionary_file)
        config.set_string("-logfn", os.devnull)  # disable logging (logging causes unwanted output in terminal)
        decoder = pocketsphinx.Decoder(config)
        return decoder

    def recognize_sphinx(self, audio_data):
        try:
            raw_data = audio_data.get_raw_data(convert_rate=16000, convert_width=2)  # the included language models require audio to be 16-bit mono 16 kHz in little-endian format

            decoder = self.decoder_queue.get()
            decoder.start_utt()  # begin utterance processing
            decoder.process_raw(raw_data, False, True)  # process audio data with recognition enabled (no_search = False), as a full utterance (full_utt = True)
            decoder.end_utt()  # stop utterance processing

            hypothesis = decoder.hyp()
        except Exception as e:
            print(e)
        finally:
            self.decoder_queue.put(decoder)

        if hypothesis is not None:
            return hypothesis.hypstr
        else:
            return ''

    """视频ASR"""
    def request(self, path):
        with sr.AudioFile(path) as source:
            audio = self.r.record(source)
        return self.recognize_sphinx(audio)
    
class VideoOCR():
    def __init__(self, use_gpu):
        self.reader = easyocr.Reader(['ch_sim'], gpu = use_gpu)
    """视频OCR"""
    def request(self, rgb):
        return list(self.reader.readtext(rgb, detail=0))
    
class ImageOCR():
    def __init__(self, use_gpu):
        self.reader = easyocr.Reader(['ch_sim'], gpu = use_gpu)

    """图像OCR"""
    def request(self, file_name):
        res = self.reader.readtext(file_name, detail = 0)
        res = set(res)
        return '|'.join(list(res))
   
if __name__ == '__main__':
    test_image = './test.jpg'
    image_ocr = ImageOCR().request(test_image)
    print("image_ocr: {}".format(image_ocr))

