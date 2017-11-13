'''
CNN.py contrains code of training and testing of the CNN module

'''

# import data_processing
import numpy as np
import os
import scipy.io
# import matplotlib.pyplot as plt
# from PIL import Image
# from keras.callbacks import EarlyStopping
# from sklearn.utils import shuffle
# from sklearn.model_selection import train_test_split
# from keras.constraints import maxnorm
# from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt
# from scipy.misc import toimage
# from keras.preprocessing import sequence
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


def run_CNN(set, result_dir):

    '''

    # :param set: set contains training, validation and testing set
                 set = {}
    #        result_dir: dir of saving the result

    :return: model : model object

    :output:
            save prediction in .mat and .png

    '''

    train_set = set['train']
    validate_set = set['validation']
    test_set = set['test']

    train_set_x = [train[0] for train in train_set]
    train_set_y = [train[1] for train in train_set]
    validate_set_x = [validate[0] for validate in validate_set]
    validate_set_y = [validate[1] for validate in validate_set]
    test_set_x = [test[0] for test in test_set]
    test_set_y = [test[1] for test in test_set]

    train_set_x = np.array(train_set_x)
    train_set_y = np.array(train_set_y)
    validate_set_x = np.array(validate_set_x)
    validate_set_y = np.array(validate_set_y)
    test_set_x = np.array(test_set_x)
    test_set_y = np.array(test_set_y)


    # build model
    model = Sequential()
    model.add(Convolution1D(filters=64, kernel_size=7, strides=1, activation='relu', input_shape=(train_set_x[0].shape[0],
                            train_set_x[0].shape[1])))
    model.add(MaxPooling1D(pool_size=2))
    # model.add(Dropout(0.2))
    # model.add(
    #     Convolution1D(filters=64, kernel_size=7, strides=1, activation='relu', input_shape=(train_set_x[0].shape[0],
    #                                                                                         train_set_x[0].shape[1])))
    # model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(1, activation='linear'))
    model.add(Dropout(0.2))
    model.add(Activation('linear'))
    model.add(Dense(1, activation='linear'))

    adam = Adam(lr=0.0001)
    model.compile(loss='mse', optimizer=adam, metrics=['mean_squared_error'])

    # training
    model.fit(train_set_x, train_set_y, nb_epoch=200, batch_size=32, validation_data=(validate_set_x, validate_set_y))

    # testing
    # on validation set
    predict_validation = model.predict(validate_set_x)
    # note: predict has shape (sample, 1)
    predict_validation = np.reshape(predict_validation, (predict_validation.shape[0],))

    output_result(name='validation', signals={'prediction':predict_validation}, truth=validate_set_y,
                  sub_folder='validation', result_dir= result_dir)

    #callback

    # on test set
    predict_test = model.predict(test_set_x)
    predict_test = np.reshape(predict_test, (predict_test.shape[0],))

    output_result(name='test', signals={'prediction': predict_test}, truth=test_set_y,
                  sub_folder='test', result_dir=result_dir)

    # return
    return model

def output_result(name, signals, truth, sub_folder, result_dir):
    # time_index : 1-D array
    # truth : 1-D array
    # signals : dictionary, 'name' : 1-D array
    save_dir = result_dir + sub_folder + '/'
    try:
        os.makedirs(save_dir)
    except:
        pass

    # compute MSE against the truth

    MSE_dic = {}
    for k in signals.keys():
        signal = signals[k]
        MSE = ((signal - truth) ** 2).sum() / len(truth)
        MSE_dic[k] = MSE

    # draw the graph
    # signals['compare'] = signals['compare'] * 0

    fig = plt.figure()
    line_truth = plt.plot(truth, label='truth')

    for k in signals.keys():
        legend = k + ' MSE=' + str(MSE_dic[k])
        line = plt.plot( signals[k], label=legend)
    plt.legend()

    try:
        plt.savefig(save_dir + 'Image/' + name + '.png')
    except:
        os.mkdir(save_dir + 'Image/')
        plt.savefig(save_dir + 'Image/' + name + '.png')
    plt.close(fig)

    # save .mat
    mat_save_dic = signals
    mat_save_dic['truth'] = truth

    try:
        scipy.io.savemat(save_dir + 'MAT/' + name + '.mat', mat_save_dic)
    except:
        os.mkdir(save_dir + 'MAT/')
        scipy.io.savemat(save_dir + 'MAT/' + name + '.mat', mat_save_dic)



