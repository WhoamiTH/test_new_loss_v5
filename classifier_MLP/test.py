import sys
from sklearn.externals import joblib
from time import clock
import handle_data
import predict_test
import numpy as np
import csv
import re
import handle_data
import predict_test
import pandas as pd
import numpy as np

import sklearn.metrics as skmet
# import tensorflow as tf

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

import torch
from torch import nn
from torch.nn import init
from sklearn.metrics import confusion_matrix

import os


def get_test_ref_samples(test_data, reference_data, test_ref_num):
    length_test_data = test_data.shape[0]
    length_reference_data = reference_data.shape[0]

    # repeat 每一个都连续重复
    test_samples = np.repeat(test_data, test_ref_num, axis=0)
    # reference data
    ref_index_list = []
    for epoch in range(length_test_data):
        ref_index_list.append( np.random.choice(length_reference_data, test_ref_num, replace=False) )
    ref_index = np.array(ref_index_list).flatten()

    ref_samples = reference_data[ref_index]
    return test_samples, ref_samples


def transform_data_to_test_form_minus(test_data, reference_data, test_ref_num):
    test_samples, ref_samples = get_test_ref_samples(test_data, reference_data, test_ref_num)
    transformed_test_data = test_samples - ref_samples
    return transformed_test_data


def transform_data_to_test_form_extern(test_data, reference_data, test_ref_num):
    test_samples, ref_samples = get_test_ref_samples(test_data, reference_data, test_ref_num)
    transformed_test_data = np.hstack((test_samples, ref_samples))
    return transformed_test_data


def transform_data_to_test_form(transform_method, test_data, reference_data, test_ref_num):
    if transform_method == 'normal':
        return test_data
    
    if transform_method == 'minus':
        return transform_data_to_test_form_minus(test_data, reference_data, test_ref_num)
    
    if transform_method == 'concat':
        return transform_data_to_test_form_extern(test_data, reference_data, test_ref_num)




def divide_data(Data, Label):
    positive_index = np.where(Label == 1)
    negative_index = np.where(Label == 0)

    positive = Data[positive_index[0]]
    negative = Data[negative_index[0]]

    return positive, negative


def loadTrainData(file_name):
    file_data = np.loadtxt(file_name, delimiter=',')
    label = file_data[:,-1]
    data = np.delete(file_data, -1, axis=1)
    data = data.astype(np.float64)
    label = label.reshape(-1, 1)
    label = label.astype(np.int)
    return data, label



def get_test_ref_num(dataset, ref_num_type, ref_times):
    ref_times = int(ref_times)
    if ref_num_type == 'num':
        return ref_times
    if ref_num_type == 'IR':
        dataset_dict = {
            'pima': 2,
            'glass0': 3,
            'vehicle0': 4,
            'ecoli1': 4,
            'yeast3': 9,
            'pageblocks1': 16,
            'glass5': 23,
            'yeast5': 33,
            'yeast6': 40,
            'abalone19': 129
        }
        basic_num = dataset_dict[dataset]
        return basic_num * ref_times

def get_reference_data(train_data, positive_data, negative_data, ref_data_type):
    if ref_data_type == 'random':
        return train_data
    if ref_data_type == 'pos':
        return positive_data
    if ref_data_type == 'neg':
        return negative_data

def get_boundray_num(test_ref_num, boundary_type):
    if boundary_type == 'half':
        return int(test_ref_num/2)
    else:
        return int(boundary_type)

# def get_test_info(test_method):
#     test_info_list = test_method.split('_')
    
#     if len(test_info_list) == 1:
#         transform_method = test_info_list[0]
#         ref_data_type = 'random'
#         ref_num_type = 'num'
#         ref_times = '1'
#         boundary_type = '1'
    
#     if len(test_info_list) == 5:
#         transform_method, ref_data_type, ref_num_type, ref_times, boundary_type = test_info_list

#     return transform_method, ref_data_type, ref_num_type, ref_times, boundary_type
    
def set_para():
    global dataset_name
    global dataset_index
    global record_index
    global train_method
    global test_method
    global device_id

    argv = sys.argv[1:]
    for each in argv:
        para = each.split('=')
        if para[0] == 'dataset_name':
            dataset_name = para[1]
        if para[0] == 'dataset_index':
            dataset_index = para[1]
        if para[0] == 'record_index':
            record_index = para[1]
        if para[0] == 'train_method':
            train_method = para[1]
        if para[0] == 'test_method':
            test_method = para[1]
        if para[0] == 'device_id':
            device_id = para[1]

# -------------------------------------parameters----------------------------------------
dataset_name = 'abalone19'
dataset_index = '1'
record_index = '1'
train_method = 'MLP_minus_mirror'
test_method = 'minus_random_num_10_half'
device_id = '1'
threshold = 0.5
# ----------------------------------set parameters---------------------------------------
set_para()

os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

train_file_name = './test_{0}/standlization_data/{0}_std_train_{1}.csv'.format(dataset_name, dataset_index)
test_file_name = './test_{0}/standlization_data/{0}_std_test_{1}.csv'.format(dataset_name, dataset_index)


# ############################# 注意修改模型及方法名称 ##############################
model_name = './test_{0}/model_{1}/record_{2}/{1}_{3}'.format(dataset_name, train_method, record_index, dataset_index)
result_record_path = './test_{0}/result_{1}_{2}/record_{3}/'.format(dataset_name, train_method, test_method, record_index)
record_name_prefix = '{0}_{1}_pred_result'.format(dataset_name, dataset_index)

print(train_file_name)
print(test_file_name)
print(train_method)
print(test_method)
print(model_name)
print(result_record_path)
print('----------------------\n\n\n')

# 在测试方法中，用 _ 分割了不同的测试部分
# transform_method, ref_data_type, ref_num_type, ref_times, boundary_type = get_test_info(test_method)
test_type, threshold = test_method.split('_')
threshold = float(threshold)
# test_ref_num = get_test_ref_num(dataset_name, ref_num_type, ref_times)
# boundary_number = get_boundray_num(test_ref_num, boundary_type)

# ------------- load train data and find reference data --------------------------------
# train_data, train_label = loadTrainData(train_file_name)
# positive_data, negative_data = divide_data(train_data, train_label)
# reference_data = get_reference_data(train_data, positive_data, negative_data, ref_data_type)
# # 如果 test_ref_num 超了 换成所有的 reference 数据
# test_ref_num = min(test_ref_num, reference_data.shape[0])

test_data, test_label = loadTrainData(test_file_name)
# cur_test_data = transform_data_to_test_form(transform_method, test_data, reference_data, test_ref_num)



start = clock()

class Classification(nn.Module):
    def __init__(self, input_dim):
        super(Classification, self).__init__()
        self.hidden_1 = nn.Linear(input_dim, 2*input_dim)
        self.relu = nn.ReLU()
        self.output = nn.Linear(2*input_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.hidden_1(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x

net = torch.load(model_name)

def get_MLP_predict_result(test_data, net):
    test_input_data = torch.Tensor(torch.from_numpy(test_data).float())
    # test_input_label = torch.Tensor(torch.from_numpy(test_label).float())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_input_data = test_input_data.to(device)
    # test_input_label = test_input_label.to(device)

    output = net(test_input_data)
    return output

# 获取初步结果
test_result = get_MLP_predict_result(test_data, net)
test_result = test_result.detach().to('cpu').numpy()

# test_length = test_data.shape[0]
# test_pred_results = test_result.reshape(test_length, test_ref_num)
test_pred_results = test_result.reshape(-1, 1)
test_pred_results[test_pred_results<threshold] = 0
test_pred_results[test_pred_results>=threshold] = 1

# # 获取 vote 结果
# if transform_method == 'normal':
#     general_vote_results = test_pred_results
# else:
#     general_vote_results = np.sum(test_pred_results, axis=1)
#     general_vote_results[general_vote_results<boundary_number] = 0
#     general_vote_results[general_vote_results>=boundary_number] = 1
    
# general_vote_results = general_vote_results.reshape(-1,1)

general_vote_results = test_pred_results

# 保存测试及预估结果数据
test_predict_data_result = np.hstack((test_data, general_vote_results))
np.savetxt(result_record_path+record_name_prefix+'.csv', test_predict_data_result, delimiter=',')

# 计算预估效果
true_label = test_label
predict_label = general_vote_results

precision = skmet.precision_score(y_true=true_label, y_pred=predict_label)
recall = skmet.recall_score(y_true=true_label, y_pred=predict_label)
fscore = skmet.f1_score(y_true=true_label, y_pred=predict_label)
auc = skmet.roc_auc_score(y_true=true_label, y_score=predict_label)

print("the AUC is {0}\n".format(auc))
print("the Fscore is {0}\n".format(fscore))
print("the precision is {0}\n".format(precision))
print("the recall is {0}\n".format(recall))

with open(result_record_path+record_name_prefix+'.txt','w') as record:
    record.write("the AUC is {0}\n".format(auc))
    record.write("the Fscore is {0}\n".format(fscore))
    record.write("the precision is {0}\n".format(precision))
    record.write("the recall is {0}\n".format(recall))

print('Done')