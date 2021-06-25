
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
1. Evaluation of different feature sets, feature types, and classifiers 
   on both song and speech emotion recognition.
2. A proposal of an acoustic feature set for song and speech emotion 
   recognition based on LibROSA tool.

## Codes  
### Song evaluation on different feature sets and types:

song_gemaps  
song_gemaps_hfs  
song_paa  
song_paa_hfs  
song_librosa  
song_librosa_hfs

**Evaluation of classifiers:**   
song_lstm  
song_gru  
song_cnn  
song_mlp  


### Speech:  
speech_gemaps  
speech_gemaps_hfs  
speech_paa  
speech_paa_hfs  
speech_librosa  
speech_librosa_hfs  

**Evaluation of classifiers:**  
speech_lstm  
speech_gru  
speech_cnn  
speech_mlp  

## Result:    
As in Table II in the paper

~~~
Feature	         Song		   Speech	
	            Acc      UAR      Acc	UAR
GeMAPS         0.637	0.592	0.602	0.614
GeMAPS HSF	   0.753	0.762	0.662	0.653
pAA            0.592	0.619	0.731	0.701
pyA HSF        0.736	0.761	0.658	0.620
LibROSA        0.751	0.780	0.732	0.676
LibROSA HSF	   0.820	0.813	0.774	0.781
~~~


