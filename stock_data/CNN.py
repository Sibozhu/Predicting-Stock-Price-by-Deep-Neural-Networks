'''
CNN.py contrains code of training and testing of the CNN module

'''

import data_processing
import numpy as np
import cv2
import PIL
import os, os.path
import matplotlib.pyplot as plt
from PIL import Image
from keras.callbacks import EarlyStopping
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.constraints import maxnorm
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from matplotlib import pyplot
from scipy.misc import toimage
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout
from keras import callbacks
from keras.layers import Dense, Flatten
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop, SGD,Adam
from keras.layers.convolutional import MaxPooling1D
from keras.layers.convolutional import Convolution1D
from keras.utils import np_utils
from keras import backend as k


def run_CNN(data_dir, result_dir):

    '''

    # :param set: set contains training and testing set
    #        result_dir: dir of saving the result
    :param data_dir: where the data located
           result_dir: dir of saving the result
    :return: testing_L2_error = (prediction - truth)**2
    '''

    #Loading data
    print('Loading data...')

    seq_len = 50
    norm_win = True
    filename = data_dir

    X_train, X_test, y_train, y_test = data_processing.load_data(filename, seq_len, norm_win)

    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')
    X_train = sequence.pad_sequences(X_train, maxlen=seq_len)
    X_test = sequence.pad_sequences(X_test, maxlen=seq_len)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    # build CNN model
    input_shape=X_train[0].shape

    model = Sequential()

    # model.add(Convolution1D(8, 8, input_shape=input_shape))
    #
    # model.add(MaxPooling1D(pool_size=(2)))

    model.add(Convolution1D(input_shape=(50, 1),
                            nb_filter=64,
                            filter_length=2,
                            border_mode='valid',
                            activation='relu',
                            ))
    model.add(MaxPooling1D(pool_length=2))


    model.add(Convolution1D(
                            nb_filter=64,
                            filter_length=2,
                            border_mode='valid',
                            activation='relu',
                            ))
    model.add(MaxPooling1D(pool_length=2))


    model.add(Dropout(0.25))
    model.add(Flatten())

    model.add(Dense(250))
    model.add(Dropout(0.25))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('linear'))



    ##########
    epochs = 100
    learning_rate = 0.01
    decay = learning_rate / epochs
    adam = Adam(lr=learning_rate)
    model.compile(loss='mse', optimizer='adam', metrics=["accuracy"])

    model.summary()
    model.get_config()
    model.layers[0].get_config()
    model.layers[0].input_shape
    model.layers[0].output_shape
    model.layers[0].get_weights()
    np.shape(model.layers[0].get_weights()[0])
    model.layers[0].trainable


    # training
    # hist = model.fit(X_train, y_train, batch_size=128, nb_epoch=100, verbose=1, validation_data=(X_test, y_test))
    #
    # filename = 'model_train_new.csv'
    # csv_log = callbacks.CSVLogger(filename, separator=',', append=False)
    #
    # early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='min')
    #
    # checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    #
    # # tensorboard callback
    # tensorboard_callback = callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True,
    #                                              write_grads=False, write_images=False, embeddings_freq=0,
    #                                              embeddings_layer_names=None, embeddings_metadata=None)
    #
    # callbacks_list = [csv_log, early_stopping, checkpoint, tensorboard_callback]

    # Evaluating the model

#    score = model.evaluate(to_categorical(X_test), to_categorical(y_test), verbose=0)
#     print('Test Loss:', score[0])
#     print('Test accuracy:', score[1])



    # Save our model here
    file = open(result_dir + "sp500predict.h5", 'a')
    model.save(result_dir + "sp500predict.h5")
    file.close()

    # # testing
    #
    # testing_L2_error = (prediction - truth)**2
    # return testing_L2_error

    # visualize and save result/ Kieran will do this part

run_CNN('./sp500_Apple_Jan2005toDec2016.csv',"./")