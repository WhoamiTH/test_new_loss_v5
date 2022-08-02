# -*- coding: utf-8 -*-
import sys
# sys.path.append('..')
# import sklearn.svm as sksvm
# import sklearn.linear_model as sklin
# import sklearn.tree as sktree
# from sklearn.externals import joblib
import joblib
import time
import sklearn.preprocessing as skpre
from sklearn.decomposition import PCA
# import handle_data
# import predict_test
import pandas as pd
import numpy as np



def loadTrainData(file_name):
    # file_data = pd.read_csv(file_name)
    file_data = np.loadtxt(file_name, delimiter=',')
    # data = file_data.values
    label = file_data[:,-1]
    # data = np.delete(data, 0, axis=1)
    data = np.delete(file_data, -1, axis=1)
    # data = np.delete(data, 0, axis=1)
    data = data.astype(np.float64)
    label = label.reshape(-1, 1)
    label = label.astype(np.int)
    # data = pd.DataFrame(data)
    return data, label


def set_para():
    global dataset_name
    global dataset_index

    argv = sys.argv[1:]
    for each in argv:
        para = each.split('=')
        if para[0] == 'dataset_name':
            dataset_name = para[1]
        if para[0] == 'dataset_index':
            dataset_index = para[1]

# -------------------------------------parameters----------------------------------------
dataset_name = 'abalone19'
dataset_index = '1'


# ----------------------------------set parameters---------------------------------------
set_para()
train_file_name = './test_{0}/origin_data/{0}_train_{1}.csv'.format(dataset_name, dataset_index)
test_file_name = './test_{0}/origin_data/{0}_test_{1}.csv'.format(dataset_name, dataset_index)
record_path = './test_{0}/standlization_data/'.format(dataset_name)
scaler_name = record_path + 'scaler_' + dataset_index + '.m'
record_train_file_name = record_path + '{0}_std_train_{1}.csv'.format(dataset_name, dataset_index)
record_test_file_name = record_path + '{0}_std_test_{1}.csv'.format(dataset_name, dataset_index)

# ----------------------------------start processing-------------------------------------
print(train_file_name)
print(test_file_name)
print(record_path)
print(scaler_name)
print(record_train_file_name)
print(record_test_file_name)
print('----------------------\n\n\n')



train_data, train_label = loadTrainData(train_file_name)
scaler = skpre.StandardScaler()
scaler.fit(train_data)
train_data = scaler.transform(train_data)
std_train_data = np.hstack((train_data, train_label))
np.savetxt(record_train_file_name, std_train_data, delimiter=',')
joblib.dump(scaler, scaler_name)

test_data, test_label = loadTrainData(test_file_name)
test_data = scaler.transform(test_data)
std_test_data = np.hstack((test_data, test_label))
np.savetxt(record_test_file_name, std_test_data, delimiter=',')
