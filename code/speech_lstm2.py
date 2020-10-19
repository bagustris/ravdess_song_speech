#!/usr/bin/env python3 

# load needed modules
import numpy as np
from keras.models import Sequential  
from keras.layers import Dense, Activation, Flatten, CuDNNLSTM, Flatten 
from keras.layers import Dropout, BatchNormalization, Bidirectional
from sklearn.model_selection import train_test_split  
from sklearn.metrics import confusion_matrix  
import pandas as pd  
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

import random as rn
import tensorflow as tf

np.random.seed(123)
rn.seed(123)
tf.set_random_seed(123)

# load feature data
X=np.load('X.npy')  
y=np.load('y.npy')
X = X.reshape((X.shape[0], 1, X.shape[1]))
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

# DNN layer units
n_dim = train_x.shape[2]  
n_classes = train_y.shape[1]  

earlystop = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)

# function to define model
def create_model():  
    model = Sequential()
    model.add(BatchNormalization(axis=-1, input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(CuDNNLSTM(n_dim, return_sequences=True))  
    model.add(CuDNNLSTM(n_dim*2, return_sequences=True))
    model.add(CuDNNLSTM(n_dim, return_sequences=True))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(n_classes, activation='softmax'))
              
    # model compilation  
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])  
    return model
   
# create the model  
model = create_model()
print(model.summary())

# train the model  
hist = model.fit(train_x, train_y, epochs=500, batch_size=32, 
                 validation_data=[test_x[:150], test_y[:150]]) #, callbacks=[earlystop])
print(max(hist.history['accuracy']), max(hist.history['val_accuracy']))
# evaluate model, test data may differ from validation data
evaluate = model.evaluate(test_x[150:], test_y[150:], batch_size=32)
print(evaluate)

