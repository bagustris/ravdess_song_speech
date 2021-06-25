# extract mean+std from 34 pyaudioanalysis feature
from keras.preprocessing import sequence
from pyAudioAnalysis import audioBasicIO 
from pyAudioAnalysis import ShortTermFeatures
import glob
import os
import numpy as np

# set data location, either speech or song
data = ''
files = glob.glob(os.path.join('./Actor_??/', '*.wav'))  
files.sort()

feat_lld = []
feat_hfs = []

for f in files: 
    print("Process...", f) 
    [Fs, x] = audioBasicIO.read_audio_file(f) 
    # only extract mono, if stereo than x[:,0] should works
    if np.ndim(x) == 1:
        F, f_names = ShortTermFeatures.feature_extraction(x, Fs, 0.025*Fs, 0.010*Fs) 
    else:
        F, f_names = ShortTermFeatures.feature_extraction(x[:,0], Fs, 0.025*Fs, 0.010*Fs) 
    mean = np.mean(F.T, axis=0) 
    std = np.std(F.T, axis=0) 
    mean_std = np.hstack([mean, std])
    #feat_lld.append(F.T) 
    feat_hfs.append(mean_std) 

#feat_np = np.array(feat)
feat_lld = np.array(feat_lld)
feat_hfs = np.array(feat_hfs)

#feat_lld = sequence.pad_sequences(feat_lld, dtype='float64')
np.save('../data/song_paa_lld.npy', feat_lld)
np.save('../data/song_paa_hsf.npy', feat_hfs)

