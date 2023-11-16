#!/usr/bin/env python3
import librosa 
import os 
import re 
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE


dir = "../patrec-files/pr_lab1/pr_lab1_2020-21_data/digits"

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
num_samples = len(wav)

for i in range(5):
    wavs, sr = wav[i] 
    print("Waveform: " + str(wavs[:3])) #plotting only the first 3 values for each waveform
    print("Speaker: " + str(speaker[i]))
    print("Digit: " + str(digit[i]))

#step 3
def extract_features(wavs):
    
    mfccs = []
    deltas = []
    delta_deltas = []
    mfscs = []

    window_length_ms = 25
    hop_step_ms = 10
    sr = 16000
    
    hop_length = int(hop_step_ms * sr / 1000)
    n_fft = int(window_length_ms*sr/1000)

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
# Filter and separate the MFCCs for digits 7 and 8 
mfccs_7 = [mfcc for mfcc, d in zip(mfccs, digit) if d == 'seven']
mfccs_8 = [mfcc for mfcc, d in zip(mfccs, digit) if d == 'eight']
mfscs_7 = [mfsc for mfsc, d in zip(mfscs, digit) if d == 'seven'][:2] #keeping only the first 2 
mfscs_8 = [mfsc for mfsc, d in zip(mfscs, digit) if d == 'eight'][:2]

# Function to create and display a histogram for a specific MFCC and digit
def plot_mfcc_histogram(mfcc_data, mfcc_number, digit):
    plt.hist(mfcc_data, bins='auto')
    plt.title(f'Histogram of {mfcc_number} MFCC for Digit {digit}')
    plt.show()

#Plot histograms for 1st and 2nd MFCCs of digit 7 
plot_mfcc_histogram([mfcc[0] for mfcc in mfccs_7], '1st', 'seven')
plot_mfcc_histogram([mfcc[1] for mfcc in mfccs_7], '2nd', 'seven')

# Plot histograms for 1st and 2nd MFCCs of digit 8
plot_mfcc_histogram([mfcc[0] for mfcc in mfccs_8], '1st', 'eight')
plot_mfcc_histogram([mfcc[1] for mfcc in mfccs_8], '2nd', 'eight')

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

#Plot correlation for MFSCs of digit 7, speaker 1 and 2
plot_correlation(mfscs_7[0], '1', 'seven')
plot_correlation(mfscs_7[1], '2', 'seven')

#Plot correlation for MFSCs of digit 8, speaker 1 and 2
plot_correlation(mfscs_8[0], '1', 'eight')
plot_correlation(mfscs_8[1], '2', 'eight')

#Plot correlation for MFCCs of digit 7, speaker 1 and 2
plot_correlation(mfccs_7[:2][0], '1', 'seven', MFSCs = False)
plot_correlation(mfccs_7[:2][1], '2', 'seven', MFSCs = False)

#Plot correlation for MFCCs of digit 8, speaker 1 and 2
plot_correlation(mfccs_8[:2][0], '1', 'eight', MFSCs = False)
plot_correlation(mfccs_8[:2][1], '2', 'eight', MFSCs = False)

#Step 5

mfccs = [mfccs[i].T for i in range(num_samples)] #resize to 133xNx13 where N = number_of_samples of each wav
deltas = [deltas[i].T for i in range(num_samples)]
delta_deltas = [delta_deltas[i].T for i in range(num_samples)]
con_vector = [np.concatenate((mfccs[i], deltas[i], delta_deltas[i]), axis=1) for i in range(num_samples)] #size = 133xNx(13x3)

mean = [np.mean(con_vector[j], axis=0) for j in range(num_samples)] #size = 133x39 
std = [np.std(con_vector[j], axis=0) for j in range(num_samples)] 

vector_data = np.concatenate((mean,std), axis=1) #size = 133x78

def scatter_plot(X_data, Y_data):
    Y_data = np.array(Y_data)
    X0, X1 = X_data[:,0], X_data[:,1] #take the first 2 dimensions 
    colors = ['pink', 'brown', 'red', 'blue', 'green', 'yellow', 'purple', 'gray', 'black']
    fig, ax = plt.subplots()
    string_label = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    i = 1
    for i, label in enumerate(string_label):
        ax.scatter(
            X0[Y_data == label], X1[Y_data == label],
            c=colors[i], label=label, 
            s=50, alpha=0.9, edgecolors='k'
        )
    ax.set_xlabel('X0')
    ax.set_ylabel('X1')
    ax.set_title('Two dimensions illustrated')
    ax.legend()
    plt.show()

scatter_plot(vector_data, digit)

#Step 6
pca_2d = PCA(n_components=2)
vector_data_2d = pca_2d.fit_transform(vector_data)
scatter_plot(vector_data_2d, digit)

def scatter_plot_3d(X_data, Y_data):
    Y_data = np.array(Y_data)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X0, X1, X2 = X_data[:, 0], X_data[:, 1], X_data[:, 2]
    colors = ['pink', 'brown', 'red', 'blue', 'green', 'yellow', 'purple', 'gray', 'black']
    string_label = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    for i, label in enumerate(string_label):
        Z0 = X0[Y_data == label]
        ax.scatter(
            X0[Y_data == label], X1[Y_data == label], X2[Y_data == label],
            c=colors[i], label=label,  
            s=30, alpha=0.9, edgecolors='k'
        )
    ax.set_xlabel('X0')
    ax.set_ylabel('X1')
    ax.set_zlabel("X2")
    ax.set_title('Three dimensions illustrated')
    ax.legend()
    plt.show()

pca_3d = PCA(n_components=3)
vector_data_3d = pca_3d.fit_transform(vector_data)
scatter_plot_3d(vector_data_3d, digit)
print(pca_2d.explained_variance_ratio_)
print(pca_3d.explained_variance_ratio_)

# Save data to be used in scripts for step 7
# scaling added to avoid variance calculations of PCA being dominated by features with higher scales
scaler = StandardScaler()
vector_data = scaler.fit_transform(vector_data)
digit = np.array(digit) #tuple to NumPy array
digit = digit.reshape(-1, 1) #reshape digits to 2D array
# concat into dataframe 
concatenated_data = np.concatenate((vector_data, digit), axis=1)
df = pd.DataFrame(concatenated_data)
# save data into csv
df.to_csv("data.csv", index=False)