# feature extraction code, bagus@ep.its.ac.id

import glob
import os
import librosa
import numpy as np

from keras.utils import to_categorical
import ntpath


# function to extract feature
def extract_feature(file_name):   
    X, sample_rate = librosa.load(file_name)  
    stft = np.abs(librosa.stft(X))  
    mfcc = np.mean(librosa.feature.mfcc(y=X, 
                                        sr=sample_rate, n_mfcc=40).T,axis=0)  
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, 
                     sr=sample_rate).T, axis=0)  
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)  
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, 
                       sr=sample_rate).T,axis=0)  
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)  
    return mfcc, chroma, mel, contrast, tonnetz  #  mfcc_delta, mfcc_delta2, 
 

# function to parse audio file given main dir and sub dir
def parse_audio_files(parent_dir, sub_dirs, file_ext="*.wav"):
    features, labels = np.empty((0, 193)), np.empty(0) 
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            try:
                print('process..', fn)
                mfcc, chroma, mel, contrast, tonnetz = extract_feature(fn)
            except Exception:
                print('cannot open', fn)
                continue
            ext_features = np.hstack([mfcc, chroma, mel, contrast, tonnetz])
            features = np.vstack([features, ext_features])
            filename = ntpath.basename(fn)
            labels = np.append(labels, filename.split('-')[2])  # grab 3rd item
    return np.array(features), np.array(labels, dtype=np.int)


main_dir = '/media/bagustris/bagus/dataset/Audio_Speech_Actors_01-24/'
sub_dir = os.listdir(main_dir)  
print("\ncollecting features and labels...")  
print("\nthis will take some time...")  
features, labels = parse_audio_files(main_dir, sub_dir)  
labels_o = to_categorical(labels)  
print("done")

np.save('X', features)
np.save('y', labels_o)
