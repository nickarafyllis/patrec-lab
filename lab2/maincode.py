#!/usr/bin/env python3
import librosa 
import os 
import re 
import matplotlib.pyplot as plt


dir = "../patrec-files/pr_lab1/pr_lab1_2020-21_data/digits"
dir = "../pr_lab1_2020-21_data/digits"

#step 2
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
    
    for filename in os.listdir(dir):
        path = os.path.join(dir, filename)
        wav.append(librosa.load(path, sr=16000))
        number, text = file_split(filename)
        digit.append(text)
        speaker.append(number)
        
    # order lists by digit and speaker number
    sorted_data = sorted(zip(wav, speaker, digit), key=lambda x: (x[2], x[1])) 
    wav, speaker, digit = zip(*sorted_data)
    
    return wav, speaker, digit

#step 3
def extract_features(wavs):
    
    mfccs = []
    deltas = []
    delta_deltas = []
    
    window_length_ms = 25
    hop_step_ms = 10
    sr=16000
    
    hop_length=int(hop_step_ms * sr / 1000)
    n_fft=int(window_length_ms*sr/1000)
    
    for wav in wavs:
        y, sr = wav 
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length, n_fft=n_fft)
        # Calculate deltas and delta-deltas
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        mfccs.append(mfcc)
        deltas.append(delta)
        delta_deltas.append(delta2)
    
    return mfccs, deltas, delta_deltas

wav, speaker, digit = data_parser(dir)
mfccs, deltas, delta_deltas = extract_features(wav)

#step 4
# Filter and separate the MFCCs for digits 3 and 4
mfccs_3 = [mfcc for mfcc, d in zip(mfccs, digit) if d == 'three']
mfccs_4 = [mfcc for mfcc, d in zip(mfccs, digit) if d == 'four']

# Function to create and display a histogram for a specific MFCC and digit
def plot_mfcc_histogram(mfcc_data, mfcc_number, digit):
    plt.hist(mfcc_data,  bins='auto')
    plt.title(f'Histogram of {mfcc_number} MFCC for Digit {digit}')
    plt.show()

# Plot histograms for 1st and 2nd MFCCs of digit 3
plot_mfcc_histogram([mfcc[0] for mfcc in mfccs_3], '1st', 'three')
plot_mfcc_histogram([mfcc[1] for mfcc in mfccs_3], '2nd', 'three')

# Plot histograms for 1st and 2nd MFCCs of digit 4
plot_mfcc_histogram([mfcc[0] for mfcc in mfccs_4], '1st', 'four')
plot_mfcc_histogram([mfcc[1] for mfcc in mfccs_4], '2nd', 'four')