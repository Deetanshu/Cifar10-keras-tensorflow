# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 09:48:27 2018

@author: deept
"""

from __future__ import print_function
import numpy as np
from keras.callbacks import EarlyStopping, TensorBoard
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.utils import to_categorical
from time import time
li=[0,1,2,3,1,2,1,2,3,5,4,3]
n_neurons=[32,64,128,128,1024]
dropVal=[0.25,0.25,0.5]
x=0
y=0
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
model = Sequential()
for i in li:
    if i==0:
        model.add(Conv2D(n_neurons[x], kernel_size=(3,3), activation='relu', input_shape=(32,32,3)))
        print("Input layer")
        x=x+1
    elif i==1:
        model.add(Conv2D(n_neurons[x], kernel_size=(3,3), activation='relu'))
        print("Convolution layer with ",n_neurons[x]," neurons added")
        x=x+1
    elif i==2:
        model.add(MaxPooling2D(pool_size=(2,2)))
        print("MaxPooling layer added ")
    elif i==3:
        model.add(Dropout(dropVal[y]))
        print("Dropout layer added")
        y=y+1
    elif i==4:
        model.add(Dense(n_neurons[x], activation='relu'))
        print("Dense layer with ",n_neurons[x]," neurons added")
        x=x+1
    elif i==5:
        model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0001, decay=1e-6),
              metrics=['accuracy'])
tensorboard = TensorBoard(log_dir='./Graph', histogram_freq=1, batch_size=32, write_graph=True, write_images=True)
model.fit(X_train / 255.0, to_categorical(Y_train),
          batch_size=128,
          shuffle=True,
          epochs=250,
          validation_data=(X_test / 255.0, to_categorical(Y_test)),
          callbacks=[EarlyStopping(min_delta=0.0001, patience=3), tensorboard])
scores = model.evaluate(X_test / 255.0, to_categorical(Y_test))
print('Loss: %.3f' % scores[0])
print('Accuracy: %.3f' % scores[1])