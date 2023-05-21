# -*- coding: utf-8 -*-
"""Copy of Group09_Assignment6_code_ConsonantVowel.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1AGXZTxePkIOvsGG9dI8lpeQ3YEsbqp2p
"""

from google.colab import drive
drive.mount('/content/drive')

import os
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Dropout

from keras.layers import Masking, SimpleRNN, Dense,LSTM

"""#### Pre processing

"""

def preprocessing_data(subdir):
  Dataset_alphabets= ['pa','paa', 're', 'sa', 'tA']
  path="/content/drive/MyDrive/Deep_learning/Group_9_Assignment6/CV_Data"
  length=[]
  data=[]
  label=[]
  for i in range(len(Dataset_alphabets)):
    for filenames in os.listdir(path+"/"+Dataset_alphabets[i]+"/"+subdir):
            file=os.path.join(path+"/"+Dataset_alphabets[i]+"/"+subdir, filenames)
            with open(file, 'r') as file:
                  contents = file.read()
                  values = contents.split()
                  numerical_values = [float(value) for value in values]
            length.append(len(numerical_values))
            data.append(numerical_values)
            label.append(i)
    print(Dataset_alphabets[i] +":"+ str(i))

  return np.array(data),np.array(label),np.array(length)

"""#### storing the dataset"""

x_train,y_train,length_train=preprocessing_data(subdir="Train")
x_test,y_test,length_test=preprocessing_data(subdir="Test")

print(np.max(length_train))
print(np.max(length_test))

pad=np.max(length_train)%39
print(pad)

mask_value=5000
x_train_pad = tf.keras.utils.pad_sequences(x_train, dtype=np.float64, padding="post", value=mask_value,maxlen=3081)
x_test_pad = tf.keras.utils.pad_sequences(x_test, dtype=np.float64, padding="post", value=mask_value,maxlen=3081)

with open(f"/content/drive/MyDrive/Deep_learning/Group_9_Assignment6/Dataset/CV_Dataset/x_train.pkl", 'wb') as f:
    pickle.dump(x_train_pad, f)
with open(f"/content/drive/MyDrive/Deep_learning/Group_9_Assignment6/Dataset/CV_Dataset/y_train.pkl", 'wb') as f:
    pickle.dump(y_train, f)

with open(f"/content/drive/MyDrive/Deep_learning/Group_9_Assignment6/Dataset/CV_Dataset/x_test.pkl", 'wb') as f:
    pickle.dump(x_test_pad, f)
with open(f"/content/drive/MyDrive/Deep_learning/Group_9_Assignment6/Dataset/CV_Dataset/y_test.pkl", 'wb') as f:
    pickle.dump(y_test, f)

x_train_pad.shape

"""#### Loading the Dataset """

with open('/content/drive/MyDrive/Deep_learning/Group_9_Assignment6/Dataset/CV_Dataset/x_train.pkl', 'rb') as f:
    x_train = pickle.load(f)
with open('/content/drive/MyDrive/Deep_learning/Group_9_Assignment6/Dataset/CV_Dataset/y_train.pkl', 'rb') as f:
    y_train = pickle.load(f)

with open('/content/drive/MyDrive/Deep_learning/Group_9_Assignment6/Dataset/CV_Dataset/x_test.pkl', 'rb') as f:
    x_test = pickle.load(f)
with open('/content/drive/MyDrive/Deep_learning/Group_9_Assignment6/Dataset/CV_Dataset/y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

print(np.array(x_train).shape)
print(np.array(x_test).shape)

3081/39

x_train=x_train.reshape(1478,-1,39)
x_test=x_test.reshape(370,-1,39)

"""#### Neural Architecture

##### Simple RNN
"""

early_stopping = EarlyStopping(monitor='loss', patience=10, min_delta=0.0001)

mask_value=5000

model = Sequential()
model.add(Masking(mask_value=mask_value, input_shape=(79, 39)))
model.add(SimpleRNN(32,activation='tanh',return_sequences=True))
model.add(SimpleRNN(64,activation='tanh',return_sequences=True))
model.add(SimpleRNN(128,activation='tanh',return_sequences=True))
model.add(SimpleRNN(256,activation='tanh',return_sequences=False))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="Softmax"))

model.summary()

Adam_optimizer=Adam(learning_rate=0.0001, beta_1=0.9,beta_2=0.999, epsilon=1e-08, name="Adam")

model.compile(optimizer='Adam', loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

history = model.fit(x_train, y_train, shuffle=True,callbacks=[early_stopping],epochs=500000,validation_split=0.0)

with open(f"/content/drive/MyDrive/Deep_learning/Group_9_Assignment6/results/CV_Results/RNN_Architecure_32_64_128_256_history.pkl", 'wb') as f:
    pickle.dump(history.history, f)

model.save(f"/content/drive/MyDrive/Deep_learning/Group_9_Assignment6/results/CV_Results/RNN_Architecure_32_64_128_256_model.h5")

"""#### Results"""

with open('/content/drive/MyDrive/Deep_learning/Group_9_Assignment6/results/CV_Results/RNN_Architecure_32_64_128_256_history.pkl', 'rb') as f:
    history = pickle.load(f)

print("Total No. of Epochs to converge "+ str(len(history['accuracy']))+"\n")
print("Training Accuracy "+ str(history['accuracy'][-1]))

fig, axs = plt.subplots(1,2,figsize=(14,2),sharex=True, sharey=False)

axs[0].set_title('RNN_Architecture')
axs[0].plot(list(range(1,len(history['loss'])+1 )), history['loss'])
axs[0].set(xlabel='No. of Epochs.',ylabel= 'loss')
axs[1].set_title('RNN_Architecture')
axs[1].plot(list(range(1,len(history['accuracy'])+1 )), history['accuracy'])
axs[1].set(xlabel='No. of Epochs.',ylabel= 'Accuracy')

plt.show()

model = keras.models.load_model('/content/drive/MyDrive/Deep_learning/Group_9_Assignment6/results/CV_Results/RNN_Architecure_32_64_128_256_model.h5')

predictions = model.predict(x_test)
conf_matrix=tf.math.confusion_matrix(y_test,predictions.argmax(axis=1))

accuracy = tf.metrics.Accuracy()(y_test, predictions.argmax(axis=1))

print("Confusion matrix:\n", conf_matrix.numpy())
print("Accuracy:", accuracy.numpy())

"""#### LSTM"""

early_stopping = EarlyStopping(monitor='loss', patience=10, min_delta=0.0001)

mask_value=5000

model = Sequential()
model.add(Masking(mask_value=mask_value, input_shape=(79,39)))
model.add(LSTM(64,activation='tanh',return_sequences=True))
model.add(LSTM(128,activation='tanh',return_sequences=True))
model.add(LSTM(256,activation='tanh',return_sequences=False))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(64, activation="Softmax"))

model.summary()

Adam_optimizer=Adam(learning_rate=0.0001, beta_1=0.9,beta_2=0.999, epsilon=1e-08, name="Adam")

model.compile(optimizer='Adam', loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

history=model.fit(x_train, y_train, shuffle=True,callbacks=[early_stopping],epochs=500000,validation_split=0.0)

with open(f"/content/drive/MyDrive/Deep_learning/Group_9_Assignment6/results/CV_Results/LSTM_Architecure_64_128_256_history.pkl", 'wb') as f:
    pickle.dump(history.history, f)

model.save(f"/content/drive/MyDrive/Deep_learning/Group_9_Assignment6/results/CV_Results/LSTM_Architecure_64_128_256_model.h5")

with open('/content/drive/MyDrive/Deep_learning/Group_9_Assignment6/results/CV_Results/LSTM_Architecure_64_128_256_history.pkl', 'rb') as f:
    history = pickle.load(f)

print("Total No. of Epochs to converge "+ str(len(history['accuracy']))+"\n")
print("Training Accuracy "+ str(history['accuracy'][-1]))

fig, axs = plt.subplots(1,2,figsize=(14,2),sharex=True, sharey=False)

axs[0].set_title('LSTM_Architecture')
axs[0].plot(list(range(1,len(history['loss'])+1 )), history['loss'])
axs[0].set(xlabel='No. of Epochs.',ylabel= 'loss')
axs[1].set_title('LSTM_Architecture')
axs[1].plot(list(range(1,len(history['accuracy'])+1 )), history['accuracy'])
axs[1].set(xlabel='No. of Epochs.',ylabel= 'Accuracy')

plt.show()

model = keras.models.load_model('/content/drive/MyDrive/Deep_learning/Group_9_Assignment6/results/CV_Results/LSTM_Architecure_64_128_256_model.h5')

predictions = model.predict(x_test)
conf_matrix=tf.math.confusion_matrix(y_test,predictions.argmax(axis=1))

accuracy = tf.metrics.Accuracy()(y_test, predictions.argmax(axis=1))

print("Confusion matrix:\n", conf_matrix.numpy())
print("Accuracy:", accuracy.numpy())