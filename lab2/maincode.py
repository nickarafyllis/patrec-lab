#!/usr/bin/env python3
import librosa 
import os 
import re 
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import seaborn as sns


dir = "../patrec-files/pr_lab1/pr_lab1_2020-21_data/digits"
#dir = "../pr_lab1_2020-21_data/digits"

#step 2
#get  speaker number and digit of a file by splitting its name
def file_split(filename):
    match = re.match(r'([A-Za-z]+)(\d+)', filename)
    text, number_part = match.groups()
    number = int(number_part)
    return number, text


#get wav, speaker number and digit of every wav file in directory
def data_parser(dir):
    
    speaker = []
    digit = []
    wav = []
   
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


#check results
wav, speaker, digit = data_parser(dir)
wavs = [element[0].tolist() for element in wav] 

for i in range(5):
    print("Waveform: " + str(wavs[i][:3]))
    print("Speaker: " + str(speaker[i]))
    print("Digit: " + str(digit[i]))


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
        # Extract MFCC & MFSC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length, n_fft=n_fft)  
        mfsc = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=13, hop_length=hop_length, n_fft=n_fft)  

        # Calculate deltas and delta-deltas
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        mfscs.append(mfsc)
        mfccs.append(mfcc)
        deltas.append(delta)
        delta_deltas.append(delta2)
    
    return mfccs, mfscs, deltas, delta_deltas

mfccs, mfscs, deltas, delta_deltas = extract_features(wav)

#step 4
# Filter and separate the MFCCs for digits 3 and 4 #GIATI 3 KAI 4?
mfccs_3 = [mfcc for mfcc, d in zip(mfccs, digit) if d == 'three']
mfccs_4 = [mfcc for mfcc, d in zip(mfccs, digit) if d == 'four']
mfscs_3 = [mfsc for mfsc, d in zip(mfscs, digit) if d == 'three'][:2] #keeping only the first 2 
mfscs_4 = [mfsc for mfsc, d in zip(mfscs, digit) if d == 'four'][:2]

# Function to create and display a histogram for a specific MFCC and digit
def plot_mfcc_histogram(mfcc_data, mfcc_number, digit):
    plt.hist(mfcc_data,  bins='auto')
    plt.title(f'Histogram of {mfcc_number} MFCC for Digit {digit}')
    plt.show()

#Plot histograms for 1st and 2nd MFCCs of digit 3 
plot_mfcc_histogram([mfcc[0] for mfcc in mfccs_3], '1st', 'three')
plot_mfcc_histogram([mfcc[1] for mfcc in mfccs_3], '2nd', 'three')

# Plot histograms for 1st and 2nd MFCCs of digit 4
plot_mfcc_histogram([mfcc[0] for mfcc in mfccs_4], '1st', 'four')
plot_mfcc_histogram([mfcc[1] for mfcc in mfccs_4], '2nd', 'four')

# Function to create and display correlation of MFSCs for a utterance 
def plot_correlation(data, speaker, digit, MFSCs = True):
    data = pd.DataFrame(data.T)
    plt.imshow(data.corr())
    plt.colorbar()
    if (MFSCs): 
        plt.title(f'Correlation matrix for MFSCs of digit {digit} uttered by speaker {speaker}')
    else:
        plt.title(f'Correlation matrix for MFCCs of digit {digit} uttered by speaker {speaker}')
    plt.show()

#Plot correlation for MFSCs of digit 3, speaker 1 and 2
plot_correlation(mfscs_3[0], '1', 'three')
plot_correlation(mfscs_3[1], '2', 'three')

#Plot correlation for MFSCs of digit 4, speaker 1 and 2
plot_correlation(mfscs_4[0], '1', 'four')
plot_correlation(mfscs_4[1], '2', 'four')

#Plot correlation for MFCCs of digit 3, speaker 1 and 2
plot_correlation(mfccs_3[:2][0], '1', 'three', MFSCs = False)
plot_correlation(mfccs_3[:2][1], '2', 'three', MFSCs = False)

#Plot correlation for MFCCs of digit 4, speaker 1 and 2
plot_correlation(mfccs_4[:2][0], '1', 'four', MFSCs = False)
plot_correlation(mfccs_4[:2][1], '2', 'four', MFSCs = False)

#Step 5

vector_data = np.zeros((133, 6*13))

for i in range(133):
    vector_data[i, :13] = np.mean(mfccs[i], axis=1)
    vector_data[i, 13:26] = np.mean(deltas[i], axis=1)
    vector_data[i, 26:39] = np.mean(delta_deltas[i], axis=1)
    vector_data[i, 39:52] = np.std(mfccs[i], axis=1)
    vector_data[i, 52:65] = np.std(deltas[i], axis=1)
    vector_data[i, 65:] = np.std(delta_deltas[i], axis=1)


def scatter_plot(X_data, Y_data):
    Y_data = np.array(Y_data)
    X0, X1 = X_data[:,0], X_data[:,1]
    colors = ['pink', 'brown', 'red', 'blue', 'green', 'yellow', 'purple', 'gray', 'black', 'orange']
    fig, ax = plt.subplots()
    string_label = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    i = 0
    for label in string_label:
            ax.scatter(
            X0[Y_data == label], X1[Y_data == label],
            c=(colors[i]), label=i,
            s=50, alpha=0.9, edgecolors='k'
        )
            i = i+1 

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title('First two dimensions illustrated')
    plt.show()

scatter_plot(vector_data,digit)