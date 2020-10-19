#!/usr/bin/env python3 

# load needed modules
import numpy as np
from keras.models import Sequential  
from keras.layers import Dense, Activation, Flatten, Dropout, GRU , BatchNormalization
 
from sklearn.model_selection import train_test_split  
from sklearn.metrics import confusion_matrix  
import pandas as pd  
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# load feature data
X = np.load('X_egemaps.npy')
y = np.load('y_egemaps.npy')  

# z-score normalization
ori_aud_features = X
norm_aud_features = []
for aud_original in ori_aud_features:
    aud_original_np = np.asarray(aud_original)
    z_norm_aud = (aud_original_np - aud_original_np.mean()) / aud_original_np.std()
    norm_aud_features.append(np.around(z_norm_aud, 6))

X = np.array(norm_aud_features)

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.33, random_state=42)

# DNN layer units
n_dim = np.array(train_x).shape[2]  
n_classes = np.array(train_y).shape[1]  

## normalize data
#mean = train_x.reshape(504172, 23).mean(axis=0)
#train_x -= mean
#std = train_x.reshape(504172, 23).std(axis=0)
#train_x /= std

#test_x -= mean
#test_x /= std

## function to define model
#def create_model():  
#    model = Sequential()  
#    # layer 1  
#    model.add(BatchNormalization(axis=-1, input_shape=(523, 23)))
#    model.add(GRU(n_dim, activation='relu', #input_shape=(523, 23),
#                  dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
#    model.add(GRU(100, activation='relu', return_sequences=True, 
#                  dropout=0.2, recurrent_dropout=0.2))
#    model.add(GRU(100, activation='relu', dropout=0.2, recurrent_dropout=0.2))
#    #model.add(Dense(128, activation='relu'))
#    model.add(Dense(n_classes, activation='softmax'))  
#    # model compilation 
#    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])  
#    return model

def dense_model(activation_function='relu', init_type='normal', optimiser='adam', dropout_rate=0.2):
    model = Sequential()   
    # layer 1  
    model.add(BatchNormalization(axis=-1, input_shape=(523, 23)))
    model.add(Flatten())
    model.add(Dense(100, kernel_initializer=init_type, activation=activation_function)) #, input_shape=(523, 23)))  
    # layer 2  
    model.add(Dense(200, kernel_initializer=init_type, activation=activation_function))  
    model.add(Dropout(dropout_rate))  
    # layer 3  
    model.add(Dense(100, kernel_initializer=init_type, activation=activation_function))  
    model.add(Dropout(dropout_rate))  
    #layer4  
    #model.add(Dense(50, kernel_initializer=init_type, activation=activation_function))  
    #model.add(Dropout(dropout_rate))
    #model.add(Flatten())
    # output layer  
    model.add(Dense(n_classes, kernel_initializer=init_type, activation='softmax'))  
    # model compilation  
    model.compile(loss='categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])  
    return model

# create the model  
model = dense_model()  
print(model.summary())

# train the model  
hist = model.fit(train_x, train_y, epochs=300, validation_data=[test_x, test_y], batch_size=32)

# evaluate model, test data may differ from validation data
evaluate = model.evaluate(test_x, test_y, batch_size=32)
print(evaluate)

