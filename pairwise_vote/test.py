# -*- coding: utf-8 -*-
import sys
sys.path.append('..')
# from sklearn.externals import joblib
# from time import clock
import joblib
import time
import handle_data
import predict_test
import numpy as np
import csv
import re
import pandas as pd
import sklearn.metrics as skmet


def set_para():
    global train_file_name
    global test_file_name
    global model_record_path
    global file_record_path
    global method_name

    global scaler_name
    global kernelpca_name
    global pca_name
    global model_name
    global record_name

    argv = sys.argv[1:]
    for each in argv:
        para = each.split('=')
        if para[0] == 'train_file_name':
            train_file_name = para[1]
        if para[0] == 'test_file_name':
            test_file_name = para[1]
        if para[0] == 'model_record_path':
            model_record_path = para[1]
        if para[0] == 'file_record_path':
            file_record_path = para[1]
        if para[0] == 'method_name':
            method_name = para[1]

        if para[0] == 'scaler_name':
            scaler_name = para[1]
        if para[0] == 'kernelpca_name':
            kernelpca_name = para[1]
        if para[0] == 'pca_name':
            pca_name = para[1]
        if para[0] == 'model_name':
            model_name = para[1]
        if para[0] == 'record_name':
            record_name = para[1]

    if kernelpca_name and pca_name:
        kernelpca_name = ''




# -------------------------------------global parameters-------------------------------
# file_name = 'NBA_test.csv'
# train_file_name = 'NBA_train.csv'
train_file_name = '../1_year_data/abalone19_train_1.csv'

# train file is the file which contained the reference data
test_file_name = '../1_year_data/abalone19_test_1.csv'
model_record_path = '../1_year_result/model_1/'
file_record_path = '../1_year_result/record_1/'
# method_name = "smote"
method_name = 'pairwise'

scaler_name = 'scaler_1.m'
pca_name = ''
kernelpca_name = ''
model_name = 'model_1.m'

threshold_value = 0
record_name = 'result_1.csv'


test_time = 10
# ----------------------------------set parameters--------------------------------------
set_para()

# ----------------------------------start processing------------------------------------


scaler_name = model_record_path + method_name + '_' + scaler_name
if pca_name != '':
    pca_name = model_record_path + method_name + '_' + pca_name
if kernelpca_name != '':
    kernelpca_name = model_record_path  + method_name + '_' + kernelpca_name
model_name = model_record_path  + method_name + '_' + model_name
record_name = file_record_path + method_name + '_' + record_name


# ------------- load train data and find reference data --------------------------------
print(train_file_name)
train_data, train_label = handle_data.loadTrainData(train_file_name)

train_data = train_data.values
train_data = train_data.astype(np.float64)
train_label = train_label.reshape((-1,1))
train_label = train_label.astype(np.int)

train_data = handle_data.transform_data_by_standarize_pca(train_data, scaler_name, pca_name, kernelpca_name)

positive_data, negative_data = handle_data.divide_data(train_data, train_label)

boundary_number = int(test_time/2)
# boundary_number = 1

# ------------- load test data and find reference data -------------------------
print(test_file_name)
file_data, test_data = handle_data.loadTestData(test_file_name)
test_data, test_label = handle_data.loadTrainData(test_file_name)
test_data = test_data.values
test_data = test_data.astype(np.float64)
test_length = test_data.shape[0]

test_data = handle_data.transform_data_by_standarize_pca(test_data, scaler_name, pca_name, kernelpca_name)

cur_test_data = handle_data.transform_data_to_test_form_data_normal(test_data, negative_data, test_time)


model = joblib.load(model_name)
# test_group_num = len(group_index_list)

test_result = model.predict(cur_test_data)
test_pred_results = test_result.reshape(test_length, -1)
test_pred_results[test_pred_results<0.5] = 0
test_pred_results[test_pred_results>=0.5] = 1

general_vote_results = np.sum(test_pred_results, axis=1)
general_vote_results[general_vote_results<boundary_number] = 0
general_vote_results[general_vote_results>=boundary_number] = 1
general_vote_results = general_vote_results.reshape(-1,1)
# all_result = []
# start = clock()

# for num in range(test_group_num):
#     print('the {0} group race'.format(num+1))
#     current_group_data = new_data[group_index_list[num]]
#     all_result += predict_test.group_test(current_group_data, model, threshold_value)
# finish = clock()

file_data['predict_result'] = general_vote_results
file_data.to_csv(record_name, index=False)



true_label = test_label
predict_label = general_vote_results

all_group_top_precision = skmet.precision_score(y_true=true_label, y_pred=predict_label)
all_group_recall = skmet.recall_score(y_true=true_label, y_pred=predict_label)
all_group_fscore = skmet.f1_score(y_true=true_label, y_pred=predict_label)
all_group_auc = skmet.roc_auc_score(y_true=true_label, y_score=predict_label)
all_group_top_exact_accuracy = 1
all_group_exact_accuracy = 1
earn_rate = 1

record_name = record_name[0:-4] + '.txt'

predict_test.cal_average(all_group_top_precision, all_group_recall, all_group_fscore, all_group_top_exact_accuracy, all_group_exact_accuracy, all_group_auc, earn_rate, record_name)



print('Done')