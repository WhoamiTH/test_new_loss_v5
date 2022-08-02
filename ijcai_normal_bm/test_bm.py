import sys
from sklearn.externals import joblib
from time import clock
import handle_data
import predict_test
import numpy as np
import csv
import re
import pandas as pd
import tensorflow as tf
import sklearn.metrics as skmet
import sampling

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# per_process_gpu_memory_fraction=0.46

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
train_file_name = 'NBA_train.csv'
# train file is the file which contained the reference data
test_file_name = 'NBA_test.csv'
model_record_path = '../1_year_result/model/'
file_record_path = '../1_year_result/record/'
method_name = "smote"

scaler_name = 'scaler.m'
kernelpca_name = ''
pca_name = ''
model_name = 'model.m'
threshold_value = 0
record_name = 'result.csv'

winner_number = 3
# test time is the reference number of test
test_time = 10

# ----------------------------------set parameters--------------------------------------
set_para()


# ----------------------------------start processing------------------------------------
print('test bm\n\n\n')

print(train_file_name)
print(test_file_name)

scaler_name = model_record_path + method_name + '_' + scaler_name
if pca_name != '':
    pca_name = model_record_path + method_name + '_' + pca_name
if kernelpca_name != '':
    kernelpca_name = model_record_path  + method_name + '_' + kernelpca_name
model_name = model_record_path  + method_name + '_' + model_name
record_name = file_record_path + method_name + '_' + record_name

print(model_name)

# ------------- load train data and find reference data --------------------------------
train_data, train_label = handle_data.loadTrainData(train_file_name)

group_index_list = handle_data.group(train_data)

train_data = train_data.values
train_data = train_data.astype(np.float64)
train_label = train_label.reshape((-1,1))
train_label = train_label.astype(np.int)

train_data = handle_data.transform_data_by_standarize_pca(train_data, scaler_name, pca_name, kernelpca_name)

positive_data, negative_data = handle_data.divide_data(train_data, train_label)


sampling_model = sampling.Sampling()
border_majority_index, informative_minority_index = sampling_model.getTestingSample(train_data, train_label)

border_majority_data = negative_data[border_majority_index]

reference_majority_length = border_majority_data.shape[0]
boundary_number = int(reference_majority_length/2)


# negative data is the reference data

# ------------- load test data and find reference data --------------------------------
file_data, test_data = handle_data.loadTestData(test_file_name)
test_data, test_label = handle_data.loadTrainData(test_file_name)
test_data = test_data.values
test_data = test_data.astype(np.float64)


start = clock()
test_data = handle_data.transform_data_by_standarize_pca(test_data, scaler_name, pca_name, kernelpca_name)


# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=per_process_gpu_memory_fraction)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess = tf.Session()

# Load the trained model from the directory "./model"
saver = tf.train.import_meta_graph(model_name+'/my_model.meta')
saver.restore(sess, tf.train.latest_checkpoint(model_name))

# Load the model components
x = tf.get_collection('x')[0]
y_true = tf.get_collection('y_true')[0]
y_pred = tf.get_collection('y_pred')[0]
cost      = tf.get_collection('cost')[0]
optimizer = tf.get_collection('optimizer')[0]




# test_dicstart, test_diclength = group(test_data)

# test_group_num = len(test_dicstart)
current_test_data = handle_data.transform_data_to_test_form_data_bm_and_im(test_data, border_majority_data)

test_result = sess.run(y_pred, feed_dict={x: current_test_data})
test_length = test_data.shape[0]
test_pred_results = test_result.reshape(test_length, -1)
test_pred_results[test_pred_results<0.5] = 0
test_pred_results[test_pred_results>=0.5] = 1
# general_pred_results = np.hstack((general_pred_results, test_pred_results))
# print(general_pred_results)

general_vote_results = np.sum(test_pred_results, axis=1)

general_vote_results[general_vote_results<boundary_number] = 0
general_vote_results[general_vote_results>=boundary_number] = 1

general_vote_results = general_vote_results.reshape(-1,1)


file_data['predict_result'] = general_vote_results
file_data.to_csv(record_name, index=False)
print('Done')


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
