# -*- coding: utf-8 -*-
import numpy as np
import random
import sklearn.preprocessing as skpre
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
# from sklearn.externals import joblib
import joblib
import pandas as pd


def loadTrainData(file_name):
    file_data = pd.read_csv(file_name)
    data = file_data.values
    label = data[:,-1]
    # data = np.delete(data, 0, axis=1)
    data = np.delete(data, -1, axis=1)
    # data = np.delete(data, 0, axis=1)
    data = data.astype(np.float64)
    data = pd.DataFrame(data)
    return data, label

# def loadTestData(file_name):
#     tem = np.loadtxt(file_name, dtype=np.str, delimiter=',')
#     column_name = tem[0,0:]
#     first_column = tem[1:,1]
#     tem_data = tem[1:, 1:]
#     data = tem_data.astype(np.float)
#     return data, column_name, first_column

def loadTestData(file_name):
    # the original data should be saved because the new result will be append to the original data to form the new data
    file_data = pd.read_csv(file_name)
    data = file_data.values
    data = np.delete(data, 0, axis=1)
    data.astype(np.float64)
    data = pd.DataFrame(data)
    return file_data, data


def divide_data(Data, Label):
    positive_index = np.where(Label == 1)
    negative_index = np.where(Label == 0)

    positive = Data[positive_index[0]]
    negative = Data[negative_index[0]]

    return positive, negative




def group(Data):
    group_index_list = []
    group_data = Data.groupby([2])
    for num, group in group_data:
        group_index_list.append(group.index.tolist())
    return group_index_list


def data_extend(Data_1, Data_2):
    m = list(Data_1)
    n = list(Data_2)
    return m + n

def condense_data_pca(Data, num_of_components):
    pca = PCA(n_components=num_of_components)
    pca.fit(Data)
    return pca


def condense_data_kernel_pca(Data, num_of_components):
    kernelpca = KernelPCA(n_components=num_of_components)
    kernelpca.fit(Data)
    return kernelpca


def standardize_data(Data):
    scaler = skpre.StandardScaler()
    scaler.fit(Data)
    return scaler


def standarize_PCA_data(Data, pca_or_not, kernelpca_or_not, num_of_components, scaler_name, pca_name, kernelpca_name):
    scaler = standardize_data(Data)
    new_data = scaler.transform(Data)
    if pca_or_not :
        pca = condense_data_pca(new_data, num_of_components)
        new_data = pca.transform(new_data)
        joblib.dump(pca, pca_name)
    if kernelpca_or_not :
        kernelpca = condense_data_kernel_pca(new_data, num_of_components)
        new_data = kernelpca.transform(new_data)
        joblib.dump(kernelpca, kernelpca_name)
    joblib.dump(scaler, scaler_name)
    return new_data

def transform_data_by_standarize_pca(Data, scaler_name, pca_name, kernelpca_name):
    scaler = joblib.load(scaler_name)
    new_data = scaler.transform(Data)
    # copy
    if pca_name:
        pca = joblib.load(pca_name)
        new_data = pca.transform(new_data)
    if kernelpca_name:
        kernelpca = joblib.load(kernelpca_name)
        new_data = kernelpca.transform(new_data)
    return new_data

def transform_data_to_test_form_data_normal(test_data, reference_data, test_time):
    temd = []
    length_test_data = test_data.shape[0]
    length_reference_data = reference_data.shape[0]

    test_index = [i for i in range(length_test_data)]
    reference_index = [i for i in range(length_reference_data)]

    test_index = np.repeat(test_index, test_time)
    reference_index = np.random.choice(reference_index, test_time*length_test_data, replace=True)

    test_sample = test_data[test_index]
    reference_samples = reference_data[reference_index]
    
    transformed_test_data = np.hstack((test_sample, reference_samples))
    return transformed_test_data  

def exchange(test_y):
    ex_ty_list = []
    rank_ty = []
    for i in range(len(test_y)):
        ex_ty_list.append((int(test_y[i]),i+1))
    exed_ty = sorted(ex_ty_list)
    for i in exed_ty:
        rank_ty.append(i[1])
    return rank_ty


def generate_primal_train_data(Data,Label,Ds,Dl,num_of_train):
    # train_index_start = random.randint(0,len(Ds)-num_of_train)
    train_index_start = 0
    front = Ds[train_index_start]
    end = Ds[train_index_start+num_of_train-1]+Dl[train_index_start+num_of_train-1]
    train_x = Data[front:end,:]
    train_y = Label[front:end]
    return train_index_start,train_x,train_y

# def handleData_extend_mirror(Data, Label, positive_value, negative_value):
#     temd = []
#     teml = []
#     length = len(Data)
#     for j in range(length):
#         for t in range(length):
#             if j != t:
#                 if Label[j] != Label[t]:
#                     temd.append(data_extend(Data[j], Data[t]))
#                     if Label[j] > Label[t]:
#                         # teml.append([-1])
#                         teml.append([negative_value])
#                     else:
#                         teml.append([positive_value])
#     return temd, teml

# def handleData_extend_not_mirror(Data, Label, positive_value, negative_value):
#     # 生成非镜像模式数据
#     temd = []
#     teml = []
#     length = len(Data)
#     for j in range(length):
#         for t in range(j+1,length):
#             if Label[j] != Label[t]:
#                 temd.append(data_extend(Data[j], Data[t]))
#                 if Label[j] > Label[t]:
#                     teml.append([positive_value])
#                 else:
#                     teml.append([negative_value])
#     return temd, teml



# def transform_data_to_compare_data(Data, Label, group_index_list, mirror_type, positive_value, negative_value):
#     tem_data = []
#     tem_label = []
#     for group_index in group_index_list:
#         current_group_data = Data[group_index]
#         current_group_label = Label[group_index]
#         if mirror_type == 'mirror':
#             temd, teml = handleData_extend_mirror(current_group_data, current_group_label, positive_value, negative_value)
#         else:
#             temd, teml = handleData_extend_not_mirror(current_group_data, current_group_label, positive_value, negative_value)
#         tem_data.extend(temd)
#         tem_label.extend(teml)

#     data = np.array(tem_data)
#     label = np.array(tem_label)

#     return data, label




def handleData_extend_mirror(positive_data, negative_data, positive_value, negative_value):
    temd = []
    teml = []
    length_pos = len(positive_data)
    length_neg = len(negative_data)
    
    for pos_index in range(length_pos):
        for neg_index in range(length_neg):
            temd.append( data_extend( positive_data[pos_index], negative_data[neg_index] ) )
            teml.append( [positive_value] )
        
            temd.append( data_extend( negative_data[neg_index], positive_data[pos_index] ) )
            teml.append( [negative_value] )
    return temd, teml


def handleData_extend_not_mirror(positive_data, negative_data, positive_value, negative_value):
    # 生成非镜像模式数据
    temd = []
    teml = []
    length_pos = len(positive_data)
    length_neg = len(negative_data)
    all_generate_num = length_pos * length_neg
    random_label_array = np.random.randint(low=0,high=2,size=all_generate_num)
    random_label_list = random_label_array.tolist()
    label_index = 0
    for pos_index in range(length_pos):
        for neg_index in range(length_neg):
            cur_label = random_label_list[label_index]
            label_index += 1
            if int(cur_label) == 1:
                temd.append( data_extend( positive_data[pos_index], negative_data[neg_index] ) )
                teml.append( [positive_value] )
            else:
                temd.append( data_extend( negative_data[neg_index], positive_data[pos_index] ) )
                teml.append( [negative_value] )
    return temd, teml

def transform_data_to_compare_data(Data, Label, mirror_type, positive_value, negative_value):
    positive_data, negative_data = divide_data(Data, Label)   

    if mirror_type == 'mirror':
        temd, teml = handleData_extend_mirror(positive_data, negative_data, positive_value, negative_value)
    else:
        temd, teml = handleData_extend_not_mirror(positive_data, negative_data, positive_value, negative_value)

    data = np.array(temd)
    label = np.array(teml)

    all_data = np.hstack( (data, label ))
    np.random.shuffle( all_data)
    
    data = all_data[:, :-1]
    label = all_data[:, -1].reshape(-1, 1)

    return data, label





def digit(x):
    if str.isdigit(x) or x == '.':
        return True
    else:
        return False

def alpha(x):
    if str.isalpha(x) or x == ' ':
        return True
    else:
        return False

def point(x):
    return x == '.'

def divide_digit(x):
    d = filter(digit, x)
    item = ''
    for i in d:
        item += i
    if len(item) == 0:
        return 0.0
    else:
        p = filter(point, item)
        itemp = ''
        for i in p:
            itemp += i
        # print(itemp)
        if len(itemp) > 1:
            return 0.0
        else:
            return float(item)

def divide_alpha(x):
    a = filter(alpha, x)
    item = ''
    for i in a:
        item += i
    return item

def divide_alpha_digit(x):
    num = divide_digit(x)
    word = divide_alpha(x)
    return word,num

def initlist():
    gp = []
    gr = []
    ga = []
    agtp = []
    agr = []
    agtea = []
    aga = []
    tt = []
    rt = []
    return gp,gr,ga,agtp,agr,agtea,aga,tt,rt

def aver(l):
    return sum(l)/len(l)

def scan_file(file_name):
    f = open(file_name,'r')
    gp,gr,ga,agtp,agr,agtea,aga,tt,rt = initlist()
    for i in f:
        word,num = divide_alpha_digit(i)
        if word == 'the average group top precision is ':
            agtp.append(num)
        if word == 'the average group recall is ':
            agr.append(num)
        if word == 'the average group top exact accuracy is ':
            agtea.append(num)
        if word == 'the average group accuracy is ':
            aga.append(num)
        if word == 'the  time training time is ':
            tt.append(float(str(num)[1:-1]))
        if word == 'the  time running time is ':
            rt.append(float(str(num)[1:-1]))
    av_aptp = aver(agtp)
    av_agr = aver(agr)
    av_agtea = aver(agtea)
    av_aga = aver(aga)
    av_tt = aver(tt)
    av_rt = aver(rt)
    return av_aptp,av_agr,av_agtea,av_aga,av_tt,av_rt

def append_file(file_name):
    av_agtp, av_agr, av_agtea, av_aga, av_tt, av_rt = scan_file(file_name)
    fscore = (2*av_agtp*av_agr)/(av_agtp+av_agr)
    f = open(file_name,'a')
    f.write("the F-score is {0}\n".format(fscore))
    f.write("the average group top precision is {0}\n".format(av_agtp))
    f.write("the average group recall is {0}\n".format(av_agr))
    f.write("the average group top exact accuracy is {0}\n".format(av_agtea))
    f.write("the average group accuracy is {0}\n".format(av_aga))
    f.write("the 3 time training time is {0}\n".format(av_tt))
    f.write("the 3 time running time is {0}\n".format(av_rt))
    f.close()



