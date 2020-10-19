#!/usr/bin/env python3 

# load needed modules
import numpy as np
from keras.models import Sequential  
from keras.layers import Dense, Activation, Flatten, Dropout, LSTM , BatchNormalization
from keras.preprocessing import sequence

from sklearn.model_selection import train_test_split  
from sklearn.metrics import confusion_matrix  
import pandas as pd  
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# load feature data
X = np.load('X_paa_old.npy')
y = np.load('y.npy')   #y_egemaps if feat_mean_std


#scaler = StandardScaler()
#X = scaler.fit_transform(X)

# z-score normalization
#ori_aud_features = X
#norm_aud_features = []
#for aud_original in ori_aud_features:
#    aud_original_np = np.asarray(aud_original)
#    z_norm_aud = (aud_original_np - aud_original_np.mean()) / aud_original_np.std()
#    norm_aud_features.append(np.around(z_norm_aud, 6))

#X = np.array(norm_aud_features)

X = sequence.pad_sequences(X, dtype=np.float64)
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.1, random_state=42)

## normalize data
#mean = train_x.reshape(504172, 23).mean(axis=0)
#train_x -= mean
#std = train_x.reshape(504172, 23).std(axis=0)
#train_x /= std

#test_x -= mean
#test_x /= std

def create_model():  
    model = Sequential()
    model.add(BatchNormalization(axis=-1, input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(LSTM(256, return_sequences=True))  
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(256, return_sequences=True))
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
#earlystop = EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)
hist = model.fit(train_x, train_y, epochs=200, 
                 batch_size=16) #, validation_split=0.1) #, callbacks=[earlystop])

# evaluate model, test data may differ from validation data
evaluate = model.evaluate(test_x, test_y, batch_size=16)
print(evaluate)

