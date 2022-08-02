#@title predict_test
# predict test head
import numpy as np
import random
import sklearn.metrics as skmet
import time
import handle_data
# transform the labels to the safe mode
def transform_labels(label, group_index_list, top_number):
    transformed_labels = []
    # circulation in groups and transform the label
    for group_index in group_index_list:
        current_true_label = label[group_index]
        current_true_label = transform_group_labels(current_true_label, top_number)
        current_true_label = list(current_true_label)
        transformed_labels.extend(current_true_label)
    transformed_labels = np.array(transformed_labels).reshape(-1,1)
    return transformed_labels


# transform the group labels to the safe mode
def transform_group_labels(label, top_number):
    length = label.shape[0]
    if length < top_number:
        label[:] = 1
        return label
    else:
        # sort_array = label.copy()
        # sort_list = list(sort_array)
        sort_list = list(label)
        sort_list = sorted(sort_list)
        for i in range(length):
            if label[i] > sort_list[top_number-1]:
                label[i] = 0
            else:
                label[i] = 1
        return label

def change_to_0_1(label,num):
    length_of_label = len(label)
    if length_of_label <= num:
        tem = [1 for t in range(length_of_label)]
    else:
        tem = [0 for t in range(length_of_label)]
        for i in range(num):
            tem[label[i]-1] = 1
    return tem


def calacc(true_label, predict_label, result_number, winner_number):
    changed_rank = change_to_0_1(true_label, winner_number)
    changed_label = change_to_0_1(predict_label,result_number)

    en = 0
    if len(changed_rank) != len(changed_label):
        print(true_label)
        print(predict_label)
        print(changed_rank)
        print(changed_label)
        time.sleep(5)
    for i in range(len(changed_rank)):
        if changed_rank[i] == changed_label[i]:
            en += 1
    return en / len(true_label)

def count_top(y_true, y_pred, result_number, winner_number):
    tp = 0
    exact = 0
    if result_number <= len(y_true):
        top_true = y_true[:winner_number]
        top_pred = y_pred[:result_number]
    elif len(y_true) >= winner_number:
        top_true = y_true[:winner_number]
        top_pred = y_pred
    else:
        top_true = y_true
        top_pred = y_pred
    len_top = len(top_pred)
    for i in range(len_top):
        if top_pred[i] in top_true:
            tp += 1
            if i < winner_number:
                if top_pred[i] == top_true[i]:
                    exact += 1
    if result_number == len_top:
        group_pre = tp/result_number
        group_recall = tp/winner_number
        group_top_exact_accuracy = exact/winner_number
    elif len_top >= winner_number:
        group_pre = tp/len_top
        group_recall = tp/winner_number
        group_top_exact_accuracy = exact/winner_number
    else:
        group_pre = tp/len_top
        group_recall = tp/len_top
        group_top_exact_accuracy = exact/len_top
    return group_pre, group_recall, group_top_exact_accuracy


def precision_recall(tp, fp, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return precision, recall


def count_general_pre(y_true, y_pred, positive_value, negative_value, threshold_value):
    tp = 0
    fp = 0
    fn = 0
    for i in range(len(y_true)):
        if int(y_true[i]) == positive_value and y_pred[i] > threshold_value:
            tp += 1
        elif int(y_true[i]) == positive_value and y_pred[i] < threshold_value:
            fn += 1
        elif int(y_true[i]) == negative_value and y_pred[i] > threshold_value:
            fp += 1
    return tp, fp, fn



# def rank_the_group(group_data, odd_data, reference, model, threshold):
#     tem = [reference.pop()]
#     for each in reference:
#         for item in range(len(tem)):
#             t = handle_data.data_extend(group_data[each-1], group_data[tem[item]-1])
#             t = np.array(t).reshape((1,-1))
#             pro_t = model.predict_proba(t)[0]
#             if (pro_t[0] * odd_data[each-1,15]) > (pro_t[1] * odd_data[tem[item]-1,15]):
#                 tem.insert(item, each)
#                 break
#             else:
#                 if item == len(tem)-1:
#                     tem.append(each)
#                     break
#     return tem

def rank_the_group(group_data, reference, model, threshold):
    tem = [reference.pop()]
    for each in reference:
        for item in range(len(tem)):
            t = handle_data.data_extend(group_data[each-1], group_data[tem[item]-1])
            t = np.array(t).reshape((1,-1))
            if model.predict(t) > threshold:
                tem.insert(item, each)
                break
            else:
                if item == len(tem)-1:
                    tem.append(each)
                    break
    return tem



def record_rank_reference(reference, rank, predict_rank, record):
    t = [i for i in range(1,len(rank)+1)]
    record_middle_result('                      ', t, record)
    record_middle_result('the random order is   ', reference, record)
    record_middle_result('the true rank is      ', rank, record)
    record_middle_result('the predict rank is   ', predict_rank, record)


def group_test(Data, model, threshold_value):
    length = len(Data)
    reference = [t for t in range(1, length + 1)]
    random.shuffle(reference)
    predict_rank = rank_the_group(Data, reference, model, threshold_value)
    predict_rank = handle_data.exchange(predict_rank)
    return predict_rank



def analyse_group_result(true_label, predict_label, result_number, winner_number, all_group_top_precision, all_group_recall, all_group_top_exact_accuracy, all_group_exact_accuracy, all_group_auc):
    group_true_label = handle_data.exchange(true_label)
    group_predict_label = handle_data.exchange(predict_label)
    group_top_precision, group_recall, group_top_exact_accuracy = count_top(group_true_label, group_predict_label, result_number, winner_number)

    group_exact_accuracy = calacc(group_true_label, group_predict_label, result_number, winner_number)

    change_true_label = change_to_0_1(group_true_label, winner_number)
    change_predict_label = change_to_0_1(group_predict_label, result_number)
    change_true_label = np.array(change_true_label)
    change_predict_label = np.array(change_predict_label)
    
    all_group_auc.append(skmet.roc_auc_score(change_true_label, change_predict_label))
    all_group_top_precision.append(group_top_precision)
    all_group_recall.append(group_recall)
    all_group_top_exact_accuracy.append(group_top_exact_accuracy)
    all_group_exact_accuracy.append(group_exact_accuracy)
    return all_group_top_precision, all_group_recall, all_group_top_exact_accuracy, all_group_exact_accuracy, all_group_auc


def cal_average(all_group_top_precision, all_group_recall, all_group_fscore, all_group_top_exact_accuracy, all_group_accuracy, all_group_auc, earn_rate, record_name):
    print("the AUC is {0}\n".format(all_group_auc))
    print("the Fscore is {0}\n".format(all_group_fscore))
    print("the average group top precision is {0}\n".format(all_group_top_precision))
    print("the average group recall is {0}\n".format(all_group_recall))
    # print("the earn rate is {0}\n".format(earn_rate))
    # print("the average group top exact accuracy is {0}\n".format(all_group_top_exact_accuracy))
    # print("the average group accuracy is {0}\n".format(all_group_accuracy))

    record = open(record_name,'w')
    record.write("the AUC is {0}\n".format(all_group_auc))
    record.write("the Fscore is {0}\n".format(all_group_fscore))
    record.write("the average group top precision is {0}\n".format(all_group_top_precision))
    record.write("the average group recall is {0}\n".format(all_group_recall))
    # record.write("the earn rate is {0}\n".format(earn_rate))
    # record.write("the average group top exact accuracy is {0}\n".format(all_group_top_exact_accuracy))
    # record.write("the average group accuracy is {0}\n".format(all_group_accuracy))

# predict test functions ending 