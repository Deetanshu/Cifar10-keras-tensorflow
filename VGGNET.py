# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 10:27:53 2018

@author: deept
"""

from __future__ import print_function
import numpy as np
from keras.callbacks import EarlyStopping,TensorBoard
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.utils import to_categorical
from keras import regularizers
#0 is input layer (convolutional in this case)
#1 is Convolutional
#2 is Maxpooling
#3 is dropout
#4 is Dense layer
#5 is flatten
li=[0,1,1,2,1,1,2,1,1,1,2,1,1,1,2,4,4,4,5,4]
#Number of neurons
n_neurons=[32,64,64,128,128,256,256,256,512,512,512,4096,4096,4096,512]
#dropout values in case if 3 is used.
dropVal=[0.25,0.25,0.5]
x=0
y=0
weight_decay = 0.0005
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
model = Sequential()
for i in li:
    if i==0:
        model.add(Conv2D(n_neurons[x], padding='same', kernel_size=(3,3), kernel_regularizer=regularizers.l2(weight_decay),activation='relu', input_shape=(32,32,3)))
        print("Input layer")
        x=x+1
    elif i==1:
        model.add(Conv2D(n_neurons[x], padding='same', kernel_size=(3,3), kernel_regularizer=regularizers.l2(weight_decay),activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        print("Convolution layer with ",n_neurons[x]," neurons added")
        x=x+1
    elif i==2:
        model.add(MaxPooling2D(pool_size=(2,2)))
        print("MaxPooling layer added ")
    elif i==3:
        model.add(Dropout(dropVal[y]))
        y=y+1
    elif i==4:
        model.add(Dense(n_neurons[x], activation='relu'))
        print("Dense layer with ",n_neurons[x]," neurons added")
        x=x+1
    elif i==5:
        model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0001, decay=1e-6),
              metrics=['accuracy'])
#Creating TensorBoard
tensorboard = TensorBoard(log_dir='./VGG', histogram_freq=0, write_graph=True, write_images=True)
model.fit(X_train / 255.0, to_categorical(Y_train),
          batch_size=128,
          shuffle=True,
          epochs=250,
          validation_data=(X_test / 255.0, to_categorical(Y_test)),
          callbacks=[EarlyStopping(min_delta=0.0001, patience=5), tensorboard])
scores = model.evaluate(X_test / 255.0, to_categorical(Y_test))
print('Loss: %.3f' % scores[0])
print('Accuracy: %.3f' % scores[1])
model.save_weights("VGGn.h5")