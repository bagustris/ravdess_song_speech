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
from sklearn.preprocessing import LabelEncoder, StandardScaler

import random as rn
import tensorflow as tf

np.random.seed(123)
rn.seed(123)
tf.set_random_seed(123)

# load feature data
X=np.load('X.npy')  
y=np.load('y.npy')

## if normalized, uncomment the following two lines
#scaler = StandardScaler()
#X = scaler.fit_transform(X)

X = X.reshape((X.shape[0], 1, X.shape[1]))

# invert labels to 1D label
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(np.argmax(y, axis=1))

# DNN layer units
n_dim = train_x.shape[2]  
n_classes = train_y.shape[1]  

earlystop = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)

# function to define model
def create_model():  
    model = Sequential()
    model.add(BatchNormalization(axis=-1, input_shape=(1, 193)))
    model.add(CuDNNLSTM(256, return_sequences=True))  
    model.add(CuDNNLSTM(256, return_sequences=True))
    model.add(CuDNNLSTM(256, return_sequences=True))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(8, activation='softmax')) #unit must match n classes
              
    # model compilation  
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])  
    return model
   
# create the model  
model = create_model()
print(model.summary())

# create model
model = KerasClassifier(build_fn=create_model, epochs=200, batch_size=16, verbose=1)

# evaluate using 5-fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())
