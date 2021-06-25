#!/usr/bin/evn python3
# python speech emotion revcognitin usin ravdess dataset

# import needed packages 
import os  
import librosa  
import numpy as np  
import pandas as pd

from keras.utils import to_categorical
from keras.preprocessing import sequence
import ntpath

import sys
sys.path.append('/media/bagustris/bagus/dataset/Audio_Audio_Actors_01-24/')

# function to parse audio file given csv file
def parse_audio_files(parent_dir):
    features = []
    hfs = [] 
    labels = np.empty(0)
    for fn in os.listdir(parent_dir):
        print('process...', fn)
        feature_i = pd.read_csv(parent_dir+fn, sep=';')
        mean_i = np.mean(feature_i.iloc[:,2:], axis=0)
        std_i = np.std(feature_i.iloc[:,2:], axis=0)
        features.append(feature_i.iloc[:,2:])
        hfs.append(np.hstack([mean_i, std_i]))
        filename = ntpath.basename(fn)
        labels = np.append(labels, filename.split('-')[2])  # grab 3rd item
    return features, hfs, np.array(labels, dtype = np.int) # 

# change directory accordingly
main_dir = '/media/bagustris/bagus/dataset/Audio_Song_Actors_01-24/audio_features_egemaps/'
print ("\ncollecting features and labels...")  
print("\nthis will take some time...")  
features, hfs, labels = parse_audio_files(main_dir)  
labels_oh = to_categorical(labels)   # one hot conversion from integer to binary

# pad features length
features = sequence.pad_sequences(features, dtype='float64')

# make sure dimension is OK
print(features.shape)
print(labels.shape)

# remove first column because label start from 1 (not from 0)
labels_oh = labels_oh[:,1:]

# If all is OK, let save it 
np.save('X_gemaps', features)
np.save('X_gemaps_hfs', hfs)
np.save('y_gemaps', labels_oh)

print("done") 
