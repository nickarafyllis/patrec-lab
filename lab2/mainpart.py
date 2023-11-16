#!/usr/bin/env python3
import librosa 
import os 
import re 
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import seaborn as sns
import parser
from sklearn.model_selection import train_test_split

dir = 'recordings/'

#step 9
X_train, X_test, y_train, y_test, spk_train, spk_test = parser.parser(dir)
X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2)





