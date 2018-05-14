import keras 
import h5py
import numpy as np
from util import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Permute,Reshape,Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

def get_cnn_model():
    model = Sequential()

    model.add(Conv2D(25, (1, 5), padding='valid',input_shape=(22,1000,1)))

    model.add(Conv2D(25, (22, 1), padding='valid'))
    #print model.layers[1].output.shape

    # 1st pooling layer
    model.add(Permute((3,2,1)))
    #print model.layers[2].output.shape

    model.add(MaxPooling2D(pool_size = (1,3)))
    
    model.add(Dropout(0.5))

    # 3rd conv layer
    model.add(Conv2D(50, (25, 10), padding='valid'))

    # 2nd pooling layer
    model.add(Permute((3,2,1)))

    model.add(MaxPooling2D(pool_size = (1,3)))

    # 4th conv layer
    model.add(Conv2D(100, (50, 10), padding='valid'))

    # 3rd pooling layer
    model.add(Permute((3,2,1)))
    model.add(MaxPooling2D(pool_size = (1,3)))
    
    # 5th conv layer
    model.add(Conv2D(200, (100, 3), padding='valid'))
    model.add(Permute((3,2,1)))
    # dense layer
    model.add(Reshape((200*30,)))
    model.add(Dense(4))

    model.add(Activation('softmax'))

    opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model

def train_cnn(model, train_x,train_y,validation_split  = 0.1):
    callbacks_list = [ModelCheckpoint('model.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='auto',     save_weights_only='True')]



    model.fit(train_x, train_y, batch_size=100, epochs=500, verbose=2, callbacks=callbacks_list,validation_split = validation_split)
    
    return model




