#!/usr/bin/env python3 

# load needed modules
import numpy as np
from keras.models import Sequential  
from keras.layers import Dense, Activation, Flatten, Dropout, LSTM , BatchNormalization
 
from sklearn.model_selection import train_test_split  
from sklearn.metrics import confusion_matrix  
import pandas as pd  
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# load feature data
X = np.load('X_paa_hfs_old.npy')
y = np.load('y.npy')   #y_egemaps if feat_mean_std

scaler = StandardScaler(with_std=False)
#X = scaler.fit_transform(X)

## z-score normalization
ori_aud_features = X
norm_aud_features = []
for aud_original in ori_aud_features:
    aud_original_np = np.asarray(aud_original)
    z_norm_aud = (aud_original_np - aud_original_np.mean()) / aud_original_np.std()
    norm_aud_features.append(np.around(z_norm_aud, 6))

X = np.array(norm_aud_features)

X = X.reshape(X.shape[0], 1, X.shape[1])
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.1)

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
#filename = os.path.basename("__file__")[:-3] +  '.svg'
plt.savefig('speech_paa_hfs.svg')
print("UAR: ", cm.trace()/cm.shape[0])
