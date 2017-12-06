from data_processing import data_process, devide
import numpy as np
from CNN_kieran import run_CNN
import os

#####Settings###############

#change the path every time
result_path = './results/'
new_dir= 'sibo16/'
#create a new directory to store new testing

try:
    os.makedirs(result_path + new_dir)
except OSError:
    if not os.path.isdir(result_path + new_dir):
        raise


result_dir = result_path+new_dir # make sure it ends with /
try:
    os.makedirs(result_dir)
except:
    pass

processed_data, name_dic = data_process('sp500_daily_full_cleaned_alter.csv')

processed_data = processed_data[:,0:405]
print('signals has length' + str(processed_data.shape[1]))
# use the first channel for test
label_signal = processed_data[[0],:]


set_list = devide(signals=processed_data, label=label_signal, input_len=30, predict_day=1, train = [0,299],
                  validation = [300, 349], test = [350, 399])

print('deviding finished!')

for i in range(len(set_list)):
    set = set_list[i]
    trained_model = run_CNN(set, result_dir + str(i) + '/')


print('debug')




