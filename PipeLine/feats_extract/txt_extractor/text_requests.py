# -*- coding: utf-8 -*-
import easyocr
import cv2
import speech_recognition as sr
import random

class VideoASR():
    def __init__(self, use_gpu):
        self.r = sr.Recognizer()

    """视频ASR"""
    def request(self, path):
        with sr.AudioFile(path) as source:
            audio = self.r.record(source)
        return self.r.recognize_sphinx(audio, language='zh-CN')
    
class VideoOCR():
    def __init__(self, use_gpu):
            self.readers = [easyocr.Reader(['ch_sim','en'], gpu = 'cuda:0'), easyocr.Reader(['ch_sim','en'], gpu = 'cuda:1')]
            #self.reader = easyocr.Reader(['ch_sim','en'], gpu = 'cuda')
    """视频OCR"""
    def request(self, rgb_list):
        index = random.randint(0, 1)
        res = []
        for rgb in rgb_list:
            t = '|'.join(self.readers[index].readtext(rgb, detail=0))
            #t = '|'.join(self.reader.readtext(rgb, detail=0))
            res.append(t)
        return res
    
