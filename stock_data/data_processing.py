'''
data processing functions
'''
import numpy as np
# import itertools
# from itertools import islice
# from operator import itemgetter
# from numpy import genfromtxt
# import pandas as pd
# # import StringIO
import csv



def data_process(data_dir):
    '''
    process and remove blank columns
    :param data_dir: .csv
    :return: 2-d numpy array, shape=(time, stock)
             name_dic = ['name_of_the_stock', ...]
    '''
    my_data = np.genfromtxt(data_dir, delimiter=',', filling_values=np.nan, usecols=range(1, 3022),skip_header=1)
    # my_data = np.recfromcsv(data_dir, delimiter=',', filling_values=np.nan, usecols=range(1,3022), case_sensitive=True, deletechars='', replace_space='')
    data_index = np.recfromcsv(data_dir, delimiter=',', filling_values=np.nan, usecols=0, case_sensitive=True, deletechars='', replace_space='')
    # my_data = np.transpose(my_data)
    # print np.shape(my_data)
    # print my_data
    # print np.shape(data_index)
    # print data_index

    return (my_data,data_index)




    # return (processed_data, name_dic)

a,b= data_process("sp500top9_Jan2005toDec2016_alter.csv")
print a
print b


def devide(signals, label, input_len, predict_day=1, train = [0,299], validation = [300, 349], test = [350, 399]):
    '''
    This function returns a list of train-validation-test sets as shifting the initial frame to the end
    it will help us to do a cross-validation-like process so that we can evaluate our performance better
    :param signals: numpy array, shape=(channels, time)
    :param label: the signal we want to predict on, numpy array, shape=(1, time)
    :param train: the first training period
    :param validation: the first period that we want to do validation on
    :param test: the first period that we want to do test on
    :return:
    input_len : length of time of input to the NN
    predict_day : 0=today, 1=1 day later

    return: set_list, list of devided datasets
    '''
    train = train
    validation = validation
    test = test

    length = label.shape[1]

    if signals.shape[1] != label.shape[1]:
        print(' signals label length mismatch!!!')
        return

    # chock
    chock_dict = {}

    start_index = 0
    end_index = start_index + input_len - 1 # end_index should be included
    label_index = end_index + predict_day

    while label_index < length:
        print('chock processing:' + str(label_index))
        x = signals[:,start_index:end_index+1]
        y = label[0, label_index]

        chock_dict[label_index] = [x,y]

        start_index += 1
        end_index += 1
        label_index +=1


    set_list = []
    while test[1] < length:
        print('set list processing:' + str(test[1]))
        train_set = []
        validation_set = []
        test_set = []
        for label_index in chock_dict.keys():
            if label_index in range(train[0], train[1] + 1):
                train_set.append(chock_dict[label_index])
            elif label_index in range(validation[0], validation[1] + 1):
                validation_set.append(chock_dict[label_index])
            elif label_index in range(test[0], test[1] + 1):
                test_set.append(chock_dict[label_index])

        set_list.append({'train': train_set, 'validation':validation_set, 'test': test_set})

        for i in range(len(train)):
            train[i] += 1
        for i in range(len(validation)):
            validation[i] += 1
        for i in range(len(test)):
            test[i] += 1

    return set_list



# def normalize_windows(win_data):
#     """
#     This function run from load_data(), and it normalize
#     data using n_i = (p_i / p_0) - 1,
#     denormalization using p_i = p_0(n_i + 1)
#     :param win_data: Window Data:
#     :return: norm_data: Normalized Window;
#     """
#     norm_data = []
#     for w in win_data:
#         norm_win = [((float(p) / float(w[0])) - 1) for p in w]
#         norm_data.append(norm_win)
#     return norm_data
#
#
# def load_data(filename, seq_len, norm_win):
#     """
#     This function loads the data from a csv file into arrays;
#     Run from CNN.py
#     :param filename: Filename
#     :param seq_len: sequence Lengh
#     :param norm_win: normalization window(True, False)
#     :return: X_tr, Y_tr, X_te, Y_te
#     """
#     fid = open(filename, 'r').read()
#     data = fid.split('\n')
#     sequence_length = seq_len + 1
#     out = []
#     for i in range(len(data) - sequence_length):
#         out.append(data[i: i + sequence_length])
#     if norm_win:
#         out = normalize_windows(out)
#         # print out
#     out = np.array(out)
#     split_ratio = 0.9
#     # print out.shape
#     split = round(split_ratio * out.shape[0])
#     # print split
#     train = out[:int(split), :]
#     np.random.shuffle(train)
#     X_train = train[:, :-1]
#     Y_train = train[:, -1]
#     X_test = out[int(split):, :-1]
#     Y_test = out[int(split):, -1]
#     X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#     X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#
#
#     return [X_train, X_test, Y_train, Y_test]
#
# load_data("sp500_Apple_Jan2005toDec2016.csv",50,True)
