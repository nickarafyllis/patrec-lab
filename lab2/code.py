#!/usr/bin/env python3
import librosa 
import os 
import numpy as np
import re 

#step 2
dir = "../pr_lab1_2020-21_data/digits"

def file_split(filename):
    match = re.match(r'([A-Za-z]+)(\d+)', filename)
    text, number_part = match.groups()
    number = int(number_part)
    return number, text

def data_parser(dir):
    wavs = []
    speaker = []
    digit = []
    wav = []
    for num, filename in enumerate(os.listdir(dir)):
        path = os.path.join(dir, filename)
        wavs.append(librosa.load(path, sr=16000))
        wav = [item[0].tolist() for item in wavs]
        number, text = file_split(filename)
        digit.append(text)
        speaker.append(number)
        break
    return wav, speaker, digit

wav, speaker, digit = data_parser(dir)


