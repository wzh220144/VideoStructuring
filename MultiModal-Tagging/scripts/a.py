#!/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2021tencent.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File Name: a.py
Author: Wang Zhihua <wangzhihua@tencent.com>
Create Time: 2021/07/05 14:35:49
Brief:      
"""

import easyocr
import cv2
import os

'''
cap = cv2.VideoCapture('/home/tione/notebook/dataset/train_5k_A/split/ad313977be6ed2ba6c8edcca59652e76#01#20.840#29.000#25.mp4')
reader = easyocr.Reader(['ch_sim'], gpu = True)
while True:
    has_frame, frame = cap.read()
    print(reader.readtext(frame, detail=0))
    if not has_frame:
        break
cap.release()
'''

'''
import speech_recognition as sr
from pocketsphinx import pocketsphinx, Jsgf, FsgModel

def get_decoder(language='zh-CN'):
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

def recognize_sphinx(audio_data, decoder):
    try:
        raw_data = audio_data.get_raw_data(convert_rate=16000, convert_width=2)  # the included language models require audio to be 16-bit mono 16 kHz in little-endian format

        decoder.start_utt()  # begin utterance processing
        decoder.process_raw(raw_data, False, True)  # process audio data with recognition enabled (no_search = False), as a full utterance (full_utt = True)
        decoder.end_utt()  # stop utterance processing

        hypothesis = decoder.hyp()
    except Exception as e:
        print(e)

    if hypothesis is not None:
        return hypothesis.hypstr
    else:
        return ''

r = sr.Recognizer()
decoder = get_decoder()

path = '/home/tione/notebook/dataset/train_5k_A/split/ad313977be6ed2ba6c8edcca59652e76#01#20.840#29.000#25.wav'
with sr.AudioFile(path) as source:
    audio = r.record(source)
    print(recognize_sphinx(audio, decoder))
'''
 
from vosk import Model, KaldiRecognizer, SetLogLevel
import sys
import os
import wave

SetLogLevel(0)

model_path = '/home/tione/notebook/VideoStructuring/pretrained/vosk-model-cn-0.1'
path = '/home/tione/notebook/dataset/train_5k_A/split/ad313977be6ed2ba6c8edcca59652e76#01#20.840#29.000#25.wav'

wf = wave.open(path, "rb")
print(wf)
'''
if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
    print ("Audio file must be WAV format mono PCM.")
    exit (1)
'''

model = Model(model_path)
rec = KaldiRecognizer(model, wf.getframerate())
rec.SetWords(True)

while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        print(rec.Result())
    else:
        print(rec.PartialResult())

print(rec.FinalResult())
