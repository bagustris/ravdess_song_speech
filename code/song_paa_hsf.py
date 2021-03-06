#!/usr/bin/env python3 

# load needed modules
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt

from keras.models import Sequential  
from keras.layers import Dense, Activation, Flatten, LSTM, Flatten 
from keras.layers import Dropout, BatchNormalization, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing import sequence

from sklearn.model_selection import train_test_split  
from sklearn.metrics import confusion_matrix  
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

import random as rn
import tensorflow as tf

np.random.seed(123)
rn.seed(123)
tf.set_random_seed(123)

# load feature data
X = np.load('../data/song_paa_hsf.npy')  
y = np.load('../data/label_gemaps.npy')

X = X.reshape(X.shape[0], 1, X.shape[1])

ori_aud_features = X
norm_aud_features = []
for aud_original in ori_aud_features:
    aud_original_np = np.asarray(aud_original)
    z_norm_aud = (aud_original_np - aud_original_np.mean()) / aud_original_np.std()
    norm_aud_features.append(np.around(z_norm_aud, 6))

X = np.array(norm_aud_features)

# invert labels to 1D label
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(np.argmax(y, axis=1))


# function to define model
def create_model():  
    model = Sequential()
    model.add(BatchNormalization(axis=-1, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(256, return_sequences=True))  
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(256, return_sequences=True))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(8, activation='softmax')) #unit must match n classes
              
    # model compilation  
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])  
    return model


# create model
#model = KerasClassifier(build_fn=create_model, epochs=200, batch_size=16, verbose=1)

## evaluate using 5-fold cross validation
#kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
#results = cross_val_score(model, X, y, cv=kfold)
#print(results.mean())

# For trying without cv
model = create_model()
print(model.summary())
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.1)
hist = model.fit(train_x, train_y, epochs=200, batch_size=16)
#evaluate = model.evaluate(test_x, test_y, batch_size=16)
#print(evaluate)

# make prediction for confusion_matrix
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns  
predict = model.predict(test_x, batch_size=16)
emotions=['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']  

# predicted emotions from the test set  
y_pred = np.argmax(predict, 1)  
predicted_emo = []   
for i in range(0,test_y.shape[0]):  
    emo = emotions[y_pred[i]]  
    predicted_emo.append(emo)

# get actual emotion
actual_emo = []  
#y_true = np.argmax(test_y, 1)  
y_true = test_y
for i in range(0,test_y.shape[0]):
    emo = emotions[y_true[i]]  
    actual_emo.append(emo)

# generate the confusion matrix  
cm = confusion_matrix(actual_emo, predicted_emo)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

#index = ['angry', 'calm', 'fearful', 'happy', 'neutral', 'sad']  
#columns = ['angry', 'calm', 'fearful', 'happy', 'neutral', 'sad']  
#cm_df = pd.DataFrame(cm, index, columns)
#plt.figure(figsize=(10, 6))  
#sns.heatmap(cm_df, annot=True)
#filename = os.path.basename(__file__)[:-3] +  '.svg'
#plt.savefig(filename)
print("UAR: ", cm.trace()/cm.shape[0])
