#!/usr/bin/env python3
import librosa 
import os 
import re 

#step 2
dir = "../patrec-files/pr_lab1/pr_lab1_2020-21_data/digits"
dir = "../pr_lab1_2020-21_data/digits"

#get  speaker number and digit of a file by splitting its name
def file_split(filename):
    match = re.match(r'([A-Za-z]+)(\d+)', filename)
    text, number_part = match.groups()
    number = int(number_part)
    return number, text

#get wav, speaker number and digit of every wav file in directory
def data_parser(dir):
    wav = []
    speaker = []
    digit = []
    wav = []
    for filename in (os.listdir(dir)):
        path = os.path.join(dir, filename)
        wav.append(librosa.load(path, sr=16000))
        number, text = file_split(filename)
        digit.append(text)
        speaker.append(number)
        
    # order lists by digit and speaker number
    sorted_data = sorted(zip(wav, speaker, digit), key=lambda x: (x[2], x[1])) 
    wav, speaker, digit = zip(*sorted_data)
    
    return wav, speaker, digit

wav, speaker, digit = data_parser(dir)

#test 
# Print all elements in the first 5 positions of each list
# print("First a wav:", wav[:1])
# print("First 5 speakers:", speaker[:5])
# print("First 5 digits:", digit[:5])