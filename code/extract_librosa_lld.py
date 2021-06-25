# extract mean+std from 34 pyaudioanalysis feature
from keras.preprocessing import sequence
from pyAudioAnalysis import audioBasicIO 
from pyAudioAnalysis import ShortTermFeatures
import glob
import os
import sys, traceback
import numpy as np

files = glob.glob(os.path.join('./Actor_??/', '*.wav'))  
files.sort()

feat_lld = []
feat_hfs = []

# import needed packages 
import glob  
import os  
import librosa  
import numpy as np  

from keras.utils import to_categorical
import ntpath

# function to extract feature
def extract_lld(file_name):   
    X, sample_rate = librosa.load(file_name)  
    stft = np.abs(librosa.stft(X))  
    mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T
    chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate).T  
    mel = librosa.feature.melspectrogram(X, sr=sample_rate).T 
    contrast = librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T  
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(X), 
                                      sr=sample_rate).T  
    return mfcc, chroma, mel, contrast, tonnetz

def extract_hfs(file_name):   
    X, sample_rate = librosa.load(file_name)  
    stft = np.abs(librosa.stft(X))  
    mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)  
    mfcc_mean = np.mean(mfccs.T, axis=0)   # 40 features, mean mfcc
    mfcc_std = np.std(mfccs.T, axis=0)
    chroma_mean = np.mean(librosa.feature.chroma_stft(S=stft, 
                                                      sr=sample_rate).T,
                                                      axis=0)  
    chroma_std = np.std(librosa.feature.chroma_stft(S=stft, 
                                                    sr=sample_rate).T,
                                                    axis=0)  
    mel_mean = np.mean(librosa.feature.melspectrogram(X, 
                                                      sr=sample_rate).T,
                                                      axis=0)  
    mel_std = np.std(librosa.feature.melspectrogram(X, 
                                                    sr=sample_rate).T, 
                                                    axis=0)  
    contrast_mean = np.mean(librosa.feature.spectral_contrast(S=stft, 
                    sr=sample_rate).T,axis=0)
    contrast_std = np.std(librosa.feature.spectral_contrast(S=stft, 
                    sr=sample_rate).T,axis=0)
    tonnetz_mean = np.mean(librosa.feature.tonnetz(
                   y=librosa.effects.harmonic(X), 
                   sr=sample_rate).T,
                   axis=0)
    tonnetz_std = np.std(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), 
                    sr=sample_rate).T,axis=0)  
    return (mfcc_mean, chroma_mean, mel_mean, contrast_mean, tonnetz_mean,
           mfcc_std, chroma_std, mel_std, contrast_std, tonnetz_std)
  
           
for fn in files: 
    print("Process...", fn) 
    try:
        print('process..', fn)
        feature_lld = extract_lld(fn)
        feature_hfs = extract_hfs(fn)
    except Exception as e:
        print('cannot open', fn)
        traceback.print_exc()
        sys.exit(3)
    
    lld_features = np.hstack(feature_lld)
    hfs_features = np.hstack(feature_hfs)
    feat_lld.append(lld_features)
    feat_hfs.append(hfs_features)

#feat_np = np.array(feat)
feat_lld = np.array(feat_lld)
feat_lld = sequence.pad_sequences(feat_lld, dtype='float64')
np.save('../data/song_librosa.npy', feat_lld)
np.save('../data/song_librosa_hfs.npy', feat_hfs)
