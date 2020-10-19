#!/usr/bin/env python3 

# load needed modules

import numpy as np
from keras.models import Sequential  
from keras.layers import Dense, Activation, Flatten, GRU, Flatten 
from keras.layers import Dropout, BatchNormalization, Bidirectional
from sklearn.model_selection import train_test_split  

import pandas as pd  
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint


import random as rn
import tensorflow as tf

np.random.seed(123)
rn.seed(123)
tf.set_random_seed(123)

# load feature data
X=np.load('X_librosa_hfs.npy')  
y=np.load('y.npy')
X = X.reshape((X.shape[0], 1, X.shape[1]))
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.1)

earlystop = EarlyStopping(monitor='val_accuracy', patience=200, restore_best_weights=True)
checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)

# function to define model
def create_model():  
    model = Sequential()
    model.add(BatchNormalization(axis=-1, input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(GRU(256, return_sequences=True))  
    model.add(GRU(256, return_sequences=True))
    model.add(GRU(256, return_sequences=True))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(8, activation='softmax'))
              
    # model compilation  
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])  
    return model
   
# create the model  
model = create_model()
print(model.summary())

# train the model  
hist = model.fit(train_x, train_y, epochs=200, shuffle=True, batch_size=16) 
evaluate = model.evaluate(test_x, test_y, batch_size=16)
print(evaluate[-1])

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
y_true = np.argmax(test_y, 1)  
for i in range(0,test_y.shape[0]):  
    emo = emotions[y_true[i]]  
    actual_emo.append(emo)

# generate the confusion matrix  
cm = confusion_matrix(actual_emo, predicted_emo)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

#index = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']  
#columns = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']  
#cm_df = pd.DataFrame(cm, index, columns)
#plt.figure(figsize=(10, 6))  
#sns.heatmap(cm_df, annot=True)
#plt.savefig('speech_librosa_hfs.svg')
print("UAR: ", cm.trace()/cm.shape[0])
