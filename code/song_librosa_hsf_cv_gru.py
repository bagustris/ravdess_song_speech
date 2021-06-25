#!/usr/bin/env python3 

# load needed modules
import numpy as np
from keras.models import Sequential  
from keras.layers import Dense, Activation, Flatten, GRU, Flatten 
from keras.layers import Dropout, BatchNormalization, Bidirectional
from sklearn.model_selection import train_test_split  
from sklearn.metrics import confusion_matrix  
import pandas as pd  
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

import random as rn
import tensorflow as tf

np.random.seed(123)
rn.seed(123)
tf.set_random_seed(123)

# load feature data
X=np.load('../data/song_librosa_hsf.npy', allow_pickle=True)  
y=np.load('../data/label_gemaps.npy', allow_pickle=True)
X = np.array([np.hstack(x) for x in X])
X = X.reshape((X.shape[0], 1, X.shape[1]))

# invert labels to 1D label
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(np.argmax(y, axis=1))

earlystop = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)# make prediction for confusion_matrix

# function to define model
def create_model():  
    model = Sequential()
    model.add(BatchNormalization(axis=-1, input_shape=(X.shape[1], X.shape[2])))
    model.add(GRU(256, return_sequences=True))  
    model.add(GRU(256, return_sequences=True))
    model.add(GRU(256, return_sequences=True))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(8, activation='softmax')) #unit must match n classes
              
    # model compilation  
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])  
    return model


# evaluate using 5-fold cross validation
# create model
# model = KerasClassifier(build_fn=create_model, 
#                         epochs=200, batch_size=16, verbose=1)

# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
# results = cross_val_score(model, X, y, cv=kfold)
# print(results.mean())

# without cross-validation
model = create_model()
print(model.summary())
train_x, test_x, train_y, test_y = train_test_split(X, y, 
                                    test_size=0.1, random_state=42)
hist = model.fit(train_x, train_y, epochs=20, batch_size=16)

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

##index = ['angry', 'calm', 'fearful', 'happy', 'neutral', 'sad']  
##columns = ['angry', 'calm', 'fearful', 'happy', 'neutral', 'sad']  
##cm_df = pd.DataFrame(cm, index, columns)
##plt.figure(figsize=(10, 6))  
##sns.heatmap(cm_df, annot=True)
##plt.savefig('song_librosa_hfs.svg')
print("Accuracy: ", accuracy_score(test_y, y_pred))
print("UAR: ", cm.trace()/cm.shape[0])

# to plot model
# plot_model(model, 'model_gru_2.pdf', show_shapes=True,  show_layer_names=False)

