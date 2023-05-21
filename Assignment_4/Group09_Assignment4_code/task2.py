# -*- coding: utf-8 -*-
"""Task2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gW5J8ApwTdo39ne-h1EsGhRP0KX27qx9
"""

from google.colab import drive
drive.mount('/content/drive')

import os
import pickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, SGD, Adagrad, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

"""#### Dataset"""

with open('/content/drive/MyDrive/Deep_learning/Group_9_Assgnment3/x_train.pkl', 'rb') as f:
    x_train = pickle.load(f)
with open('/content/drive/MyDrive/Deep_learning/Group_9_Assgnment3/y_train.pkl', 'rb') as f:
    y_train = pickle.load(f)

with open('/content/drive/MyDrive/Deep_learning/Group_9_Assgnment3/x_test.pkl', 'rb') as f:
    x_test = pickle.load(f)
with open('/content/drive/MyDrive/Deep_learning/Group_9_Assgnment3/y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

with open('/content/drive/MyDrive/Deep_learning/Group_9_Assgnment3/x_val.pkl', 'rb') as f:
    x_val = pickle.load(f)
with open('/content/drive/MyDrive/Deep_learning/Group_9_Assgnment3/y_val.pkl', 'rb') as f:
    y_val = pickle.load(f)

x_train = np.array(x_train).reshape(-1, 784).tolist()
x_test = np.array(x_test).reshape(-1, 784).tolist()
x_val = np.array(x_val).reshape(-1, 784).tolist()

"""#### 1-layer Autoencoders"""

initializer = keras.initializers.RandomUniform(minval=-1, maxval=1, seed=100)
early_stopping = EarlyStopping(monitor='loss', patience=5, min_delta=0.0001)

input_size=28*28
bottleneck_size=256

model = keras.Sequential([
    layers.Input(shape=(input_size,),name="Input_layer"),
    layers.Dense(bottleneck_size, activation='tanh',name="Bottleneck_layer"),
    layers.Dense(input_size, activation='linear',name="output")
])
model.summary()

Adam_optimizer=Adam(learning_rate=0.001, beta_1=0.9,beta_2=0.999, epsilon=1e-08, name="Adam")

model.compile(optimizer=Adam_optimizer, loss='mean_squared_error', metrics=['accuracy'])

history = model.fit(x_train, x_train, batch_size=1, epochs=500000, verbose=1, shuffle=True, callbacks=[early_stopping], validation_split=0.0)

with open(f"/content/drive/MyDrive/Deep_learning/Group_9_Assignment4/1-layer_Autoencoder/"+str(bottleneck_size)+"_1-layer_Autoencoder_history.pkl", 'wb') as f:
    pickle.dump(history.history, f)

model.save(f"/content/drive/MyDrive/Deep_learning/Group_9_Assignment4/1-layer_Autoencoder/"+str(bottleneck_size)+"_1-layer_Autoencoder_model.h5")

"""#### 3-layer Autoencoders """

initializer = keras.initializers.RandomUniform(minval=-1, maxval=1, seed=100)
early_stopping = EarlyStopping(monitor='loss', patience=5, min_delta=0.0001)

input_size=28*28
bottleneck_size=256

model = keras.Sequential([
    layers.Input(shape=(input_size,),name="Input_layer"),
    layers.Dense(400, activation='tanh', name="Encoder_Hidden_layer1"),
    layers.Dense(bottleneck_size, activation='tanh',name="Bottleneck_layer"),
    layers.Dense(400, activation='tanh',name="Decoder_Hidden_layer1"),
    layers.Dense(input_size, activation='linear',name="output")
])
model.summary()

Adam_optimizer=Adam(learning_rate=0.001, beta_1=0.9,beta_2=0.999, epsilon=1e-08, name="Adam")

model.compile(optimizer=Adam_optimizer, loss='mean_squared_error', metrics=['accuracy'])

history = model.fit(x_train, x_train, batch_size=1, epochs=500000, verbose=1, shuffle=True, callbacks=[early_stopping], validation_split=0.0)

with open(f"/content/drive/MyDrive/Deep_learning/Group_9_Assignment4/2-layer_Autoencoder/"+str(bottleneck_size)+"_2-layer_Autoencoder__history.pkl", 'wb') as f:
    pickle.dump(history.history, f)

model.save(f"/content/drive/MyDrive/Deep_learning/Group_9_Assignment4/2-layer_Autoencoder/"+str(bottleneck_size)+"_2-layer_Autoencoder_model.h5")

"""#### Results"""

autoencoder =  keras.models.load_model('/content/drive/MyDrive/Deep_learning/Group_9_Assignment4/2-layer_Autoencoder/256_2-layer_Autoencoder_model.h5')

loss,accuracy= autoencoder.evaluate(x_train, x_train, batch_size=None, verbose=1, callbacks=None)
print(f"Average reconstruction training error: {loss:.4f}")

loss,accuracy= autoencoder.evaluate(x_val, x_val, batch_size=None, verbose=1, callbacks=None)
print(f"Average reconstruction validation error: {loss:.4f}")

loss,accuracy= autoencoder.evaluate(x_test, x_test, batch_size=None, verbose=1, callbacks=None)
print(f"Average reconstruction testing error: {loss:.4f}")

def Dataset(path):
    data = []
    #label=[]
    for subdir, dirs, filenames in os.walk(path):
        for filename in filenames:
            img=np.array(Image.open(os.path.join(subdir, filename)))
            data.append(img)
            #label.append(float(subdir[-1]))
    return data #,label

train_directory="/content/drive/MyDrive/Deep_learning/Group_9_Assignment4/Train"
x_train = Dataset(train_directory)
x_train = np.array(x_train).astype("float32") / 255.0
x_train = x_train.reshape(-1, 784).tolist()

model =  keras.models.load_model('/content/drive/MyDrive/Deep_learning/Group_9_Assignment4/2-layer_Autoencoder/256_2-layer_Autoencoder_model.h5')
reconstructed_train_data = model.predict(x_train)

x_train=np.array(x_train).reshape(5,28,28)
reconstructed_train_data=np.array(reconstructed_train_data).reshape(-1,28,28)

fig, axs = plt.subplots(1, 5, figsize=(12, 12))


axs[0].imshow(x_train[0,:,:])
axs[1].imshow(x_train[1,:,:])
axs[2].set_title('Training Image')
axs[2].imshow(x_train[2,:,:])
axs[3].imshow(x_train[3,:,:])
axs[4].imshow(x_train[4,:,:])
plt.show()

fig, axs = plt.subplots(1, 5, figsize=(12,12))


axs[0].imshow(reconstructed_train_data[0,:,:])
axs[1].imshow(reconstructed_train_data[1,:,:])
axs[2].set_title('Reconstructed Image')
axs[2].imshow(reconstructed_train_data[2,:,:])
axs[3].imshow(reconstructed_train_data[3,:,:])
axs[4].imshow(reconstructed_train_data[4,:,:])
plt.show()