import keras
import h5py
import numpy as np
from util import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Permute,Reshape,Dropout,Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

def get_scnn_model():
    model = Sequential()
    model.add(Conv2D(40, (1, 25), padding='valid',input_shape=(22,1000,1)))
    model.add(Conv2D(40, (22, 1), padding='valid'))
    model.add(Permute((3,2,1)))
    model.add(MaxPooling2D(pool_size = (1,75)))
    model.add(Flatten())
    model.add(Dense(4))
    model.add(Activation('softmax'))
    
    opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model

def get_nn_model():
    model = Sequential()
    model.add(Conv2D(25,(1,5),input_shape = (22,1000,1)))
    model.add(Flatten())
    model.add(Dense(4))
    model.add(Activation('softmax'))
    
    opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model

