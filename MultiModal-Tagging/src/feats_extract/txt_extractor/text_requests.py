# -*- coding: utf-8 -*-
import easyocr
import cv2
import speech_recognition as sr

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
        self.reader = easyocr.Reader(['ch_sim','en'], gpu = use_gpu)
    """视频OCR"""
    def request(self, rgb_list):
        res = set([])
        for rgb in rgb_list:
            t = self.reader.readtext(rgb, detail=0)
            for x in t:
                res.add(x)
        return '|'.join(list(res))
    
class ImageOCR():
    def __init__(self, use_gpu):
        self.reader = easyocr.Reader(['ch_sim','en'], gpu = use_gpu)

    """图像OCR"""
    def request(self, file_name):
        res = self.reader.readtext(file_name, detail = 0)
        res = set(res)
        return '|'.join(list(res))
   
if __name__ == '__main__':
    test_image = './test.jpg'
    image_ocr = ImageOCR().request(test_image)
    print("image_ocr: {}".format(image_ocr))

