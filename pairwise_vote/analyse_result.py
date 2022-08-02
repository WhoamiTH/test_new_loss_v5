import sys
sys.path.append('..')
import handle_data
import predict_test
import glob
import re
import numpy as np
import sklearn.metrics as skmet
import pandas as pd


def set_para():
    global predict_label_file_name
    global true_label_file_name
    global model_record_path
    global file_record_path
    global method_name
    global winner_number
    global record_name


    argv = sys.argv[1:]
    for each in argv:
        para = each.split('=')
        if para[0] == 'predict_label_file_name':
            predict_label_file_name = para[1]
        if para[0] == 'true_label_file_name':
            true_label_file_name = para[1]
        if para[0] == 'model_record_path':
            model_record_path = para[1]
        if para[0] == 'file_record_path':
            file_record_path = para[1]
        if para[0] == 'method_name':
            method_name = para[1]
        if para[0] == 'winner_number':
            winner_number = int(para[1])
        if para[0] == 'record_name':
            record_name = para[1]

# -------------------------------------global parameters-------------------------------
predict_label_file_name="svm_result.csv"
true_label_file_name = "GData_test_origin.csv"
model_record_path = '../1_year_result/model/'
file_record_path = '../1_year_result/record/'
method_name = 'smote'
winner_number = 3
record_name = 'result.txt'

each_cost = 1
true_label_record_path = "../1_year_result/true/"
# ----------------------------------set parameters--------------------------------------
set_para()

# ----------------------------------start processing---------------------------------
predict_label_file_name = file_record_path + predict_label_file_name
true_label_file_name = true_label_record_path + true_label_file_name
record_name = file_record_path + method_name + '_' + record_name

print(record_name)

# load data and label
true_data, true_label = handle_data.loadTrainData(true_label_file_name)
predict_data, predict_label = handle_data.loadTrainData(predict_label_file_name)

true_label = true_label.astype(np.int)
predict_label = predict_label.astype(np.int)

group_index_list = handle_data.group(true_data)
true_label = predict_test.transform_labels(true_label, group_index_list, winner_number)
predict_label = predict_test.transform_labels(predict_label, group_index_list, winner_number)

all_group_top_precision = skmet.precision_score(y_true=true_label, y_pred=predict_label)
all_group_recall = skmet.recall_score(y_true=true_label, y_pred=predict_label)
all_group_fscore = skmet.f1_score(y_true=true_label, y_pred=predict_label)
all_group_auc = skmet.roc_auc_score(y_true=true_label, y_score=predict_label)
all_group_top_exact_accuracy = 1
all_group_exact_accuracy = 1
earn_rate = 1

predict_test.cal_average(all_group_top_precision, all_group_recall, all_group_fscore, all_group_top_exact_accuracy, all_group_exact_accuracy, all_group_auc, earn_rate, record_name)
