
On the differences between Song and Speech emotion recognition:
Effect of feature sets, feature types, and classifiers

Feature sets:
 - GeMAPS
 - pAA
 - LibROSA

Feature types:
 - LLD
 - HFS

Classifiers:  
 - LSTM
 - GRU
 - CNN
 - MLP


Contribution:  
1. Evaluation of different featute sets, feature types, and classifiers 
   on both song and speech emotion recognition.
2. A proposal of an acoustic feature set for song and speech emotion 
   recognition based on LibROSA tool.

## Codes  
song:  
song_gemaps  
song_gemaps_hfs  
song_paa  
song_paa_hfs  
song_librosa  
song_librosa_hfs

song_lstm  
song_gru  
song_cnn  
song_mlp  


speech:  
speech_gemaps  
speech_gemaps_hfs  
speech_paa  
speech_paa_hfs  
speech_librosa  
speech_librosa_hfs  

speech_lstm  
speech_gru  
speech_cnn  
speech_mlp  

## Result:    


| Method                 | accuracy     |
|------------------------|---------------|
| ser_ravdess-egemaps       | 0.5609243512153625 |  
| ser_ravdess-cudnn-lstm    | 0.6666666865348816 *|  
| ser_ravdess-cudnn-blstm   |  0.6811594367027283 |
| ser_ravdess-cudnn-lstm2   |  0.7246376872062683 **|

\* with callbacks  
** without callbacks


