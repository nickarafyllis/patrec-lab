#!/usr/bin/env python3
import numpy as np
from pomegranate.distributions import Normal
from pomegranate.gmm import GeneralMixtureModel 
from pomegranate.hmm import DenseHMM
from parser import parser
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from plot_confusion_matrix import plot_confusion_matrix
from scipy.stats import norm
import torch
import itertools
from plot_confusion_matrix import plot_confusion_matrix

# TODO: YOUR CODE HERE
# Play with diffrent variations of parameters in your experiments
n_states = 4  # the number of HMM states
n_mixtures = 3  # the number of Gaussians
gmm = True # whether to use GMM or plain Gaussian
covariance_type = "diag"  # Use diagonal covariange


# Gather data separately for each digit
def gather_in_dic(X, labels, spk):
    dic = {}
    for dig in set(labels):
        x = [X[i] for i in range(len(labels)) if labels[i] == dig]
        lengths = [len(i) for i in x]
        y = [dig for _ in range(len(x))]
        s = [spk[i] for i in range(len(labels)) if labels[i] == dig]
        dic[dig] = (x, lengths, y, s)
    return dic


def create_data():
    X, X_test, y, y_test, spk, spk_test = parser("recordings", n_mfcc=13)
    # TODO: YOUR CODE HERE
    (
        X_train,
        X_val,
        y_train,
        y_val,
        spk_train,
        spk_val,
    ) = train_test_split(X, y, spk, stratify=y, test_size=0.2)  # split X into a 80/20 train validation split
    train_dic = gather_in_dic(X_train, y_train, spk_train)
    val_dic = gather_in_dic(X_val, y_val, spk_val)
    test_dic = gather_in_dic(X_test, y_test, spk_test)
    labels = list(set(y_train))
    return train_dic, y_train, val_dic, y_val, test_dic, y_test, labels


def initialize_and_fit_gmm_distributions(X, n_states, n_mixtures):
    # TODO: YOUR CODE HERE
    dists = []
    for _ in range(n_states):
        distributions = [Normal(covariance_type) for i in range(n_mixtures)] #initializing n_mixtures gaussians
        gmm = GeneralMixtureModel(distributions, verbose=True)
        gmm.fit(np.concatenate(X))
        dists.append(gmm)
    return dists


def initialize_and_fit_normal_distributions(X, n_states):
    dists = []
    for _ in range(n_states):
        # TODO: YOUR CODE HERE
        dist = Normal(covariance_type).fit(np.concatenate(X))
        dists.append(dist)
    return dists


def initialize_transition_matrix(n_states):
    # TODO: YOUR CODE HERE
    # Make sure the dtype is np.float32
    # Initialize transition matrix with zeros
    transition_matrix = np.zeros((n_states, n_states), dtype=np.float32)

    # Set transition probabilities
    for i in range(n_states):
        for j in range(n_states):
            if j == i or j == i + 1:  # Transition to self or to next state
                transition_matrix[i, j] = 0.5  # Adjust as needed based on your specific scenario
            else:
                transition_matrix[i, j] = 0.0  # No other transitions allowed in left-right HMM

    # Normalize rows to ensure probabilities sum up to 1
    transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
    print(transition_matrix)
    return transition_matrix


def initialize_starting_probabilities(n_states):
    # TODO: YOUR CODE HERE
    # Make sure the dtype is np.float32
    starting_probabilities = np.zeros(n_states, dtype=np.float32)
    starting_probabilities[0] = 1.0  #First state has probability = 1.0
    return starting_probabilities


def initialize_end_probabilities(n_states):
    # TODO: YOUR CODE HERE
    # Make sure the dtype is np.float32
    end_probabilities = np.zeros(n_states, dtype=np.float32)
    end_probabilities[n_states - 1] = 1.0  #Last state has probability = 1.0 
    return end_probabilities


def train_single_hmm(X, emission_model, digit, n_states):
    A = initialize_transition_matrix(n_states)
    start_probs = initialize_starting_probabilities(n_states)
    end_probs = initialize_end_probabilities(n_states)
    data = [x.astype(np.float32) for x in X]

    model = DenseHMM(
        distributions=emission_model,
        edges=A,
        starts=start_probs,
        ends=end_probs,
        verbose=True,
    ).fit(data)
    return model


def train_hmms(train_dic, labels, gmm, n_mixtures, n_states):
    hmms = {}  # create one hmm for each digit

    for dig in labels:
        X, _, _, _ = train_dic[dig]
        
        # TODO: YOUR CODE HERE
        if (gmm):
            emission_model = initialize_and_fit_gmm_distributions(X, n_states, n_mixtures)
        else:
            emission_model = initialize_and_fit_normal_distributions(X, n_states)
        hmms[dig] =  train_single_hmm(X, emission_model, dig, n_states)

    return hmms


def evaluate(hmms, dic, labels):
    pred, true = [], []
    for dig in labels:
        X, _, _, _ = dic[dig]
        for sample in X:
            ev = [0] * len(labels)
            sample = np.expand_dims(sample, 0)
            for digit, hmm in hmms.items():
                # TODO: YOUR CODE HERE
                logp =  hmm.log_probability(sample) # use the hmm.log_probability function
                ev[digit] = logp
          
            # TODO: YOUR CODE HERE
            predicted_digit = max(range(len(ev)), key=lambda i: ev[i].item()) # Calculate the most probable digit
            pred.append(predicted_digit)
            true.append(dig)
    return pred, true


train_dic, y_train, val_dic, y_val, test_dic, y_test, labels = create_data()
labels = list(set(y_train))

# TODO: YOUR CODE HERE
# Calculate and print the accuracy score on the validation and the test sets
# Plot the confusion matrix for the validation and the test set

def calculate_acc(pred, true):
    correct_predictions = sum(p == t for p, t in zip(pred, true))
    return correct_predictions / len(pred)

#hmms = train_hmms(train_dic, labels, gmm, n_mixtures, n_states)

#acc_val = (calculate_acc(pred_val, true_val))
#print("states:%d, Gaussians:%d, has %f accuracy for Validation Set" %(n_states, n_mixtures, acc_val))

best_n_states = 4 
best_n_mixtures = 3  

best_hmms = train_hmms(train_dic, labels, gmm, best_n_mixtures, best_n_states,) 
pred_train, true_train = evaluate(best_hmms, train_dic, labels)
pred_val, true_val = evaluate(best_hmms, val_dic, labels)
pred_test, true_test = evaluate(best_hmms, test_dic, labels)
acc_test = calculate_acc(pred_test, true_test)
print("states:%d, Gaussians:%d, has %f accuracy for Test Set" %(best_n_states, best_n_mixtures, acc_test))

#confusion matrix
def calculate_cm(pred, true):
    cm = np.zeros((10, 10)) #initialize confusion matrix
    for i in range(len(true)):
        cm[true[i], pred[i]] += 1
    return cm

cm_test = calculate_cm(pred_test, true_test)
plt.rcParams['figure.figsize'] = [25, 20]
plot_confusion_matrix(cm_test, [i for i in range(10)], normalize=True)
cm_val = calculate_cm(pred_val, true_val)
plt.rcParams['figure.figsize'] = [25, 20]
plot_confusion_matrix(cm_val, [i for i in range(10)], normalize=True)

acc_total = calculate_acc(pred_test+pred_val+pred_train, true_test+true_val+true_train)
print("states:%d, Gaussians:%d, has %f accuracy for Total Set" %(best_n_states, best_n_mixtures, acc_total))