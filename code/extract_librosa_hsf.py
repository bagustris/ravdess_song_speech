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
    mfcc = np.mean(librosa.feature.mfcc(
        y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    mfcc_std = np.std(librosa.feature.mfcc(
        y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(
        S=stft, sr=sample_rate).T, axis=0)
    chroma_std = np.std(librosa.feature.chroma_stft(
        S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    mel_std = np.std(librosa.feature.melspectrogram(
        X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(
        S=stft, sr=sample_rate).T, axis=0)
    contrast_std = np.std(librosa.feature.spectral_contrast(
        S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(
        y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
    tonnetz_std = np.std(librosa.feature.tonnetz(
        y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
    return (mfcc, chroma, mel, contrast, tonnetz,
            mfcc_std, chroma_std, mel_std, contrast_std, tonnetz_std)


# function to parse audio file given main dir and sub dir
def parse_audio_files(parent_dir, sub_dirs, file_ext="*.wav"):
    features, labels = np.empty((0, 386)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            try:
                print('process..', fn)
                mean_std = extract_feature(fn)
            except Exception as e:
                print('cannot open', fn)
                continue
            ext_features = np.hstack(mean_std)
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

np.save('../data_d/X_librosa_hsf', features)
np.save('../data_d/y_librosa_hsf', labels_o)
