# -*- coding: utf-8 -*-

'''
    将所有数据都画在一张图里，但是会有超内存的问题，所以需要分开运算，分开画
    这个脚本只画原始数据转换后的结果
'''



import sys
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE 
# import joblib
import time
import sklearn.preprocessing as skpre


import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

def condense_data_tsne(Data, num_of_components=2):
    tsne = TSNE(n_components=num_of_components)
    new_data = tsne.fit_transform(Data) 
    return new_data

def condense_data_pca(Data, num_of_components=2):
    pca = PCA(n_components=num_of_components)
    pca.fit(Data)
    new_data = pca.transform(Data)
    return new_data

def condense_data_kernel_pca(Data, num_of_components=2):
    kernelpca = KernelPCA(n_components=num_of_components)
    kernelpca.fit(Data)
    new_data = kernelpca.transform(Data)
    return new_data

def divide_data(Data, Label):
    positive_index = np.where(Label == 1)
    negative_index = np.where(Label == 0)

    positive = Data[positive_index[0]]
    negative = Data[negative_index[0]]

    return positive, negative



def handleData_minus_mirror(positive_data, negative_data):
    # 生成镜像数据
    length_pos = positive_data.shape[0]
    length_neg = negative_data.shape[0]
    all_generate_num = length_pos * length_neg

    # repeat 每一个都连续重复
    positive_repeat_data = np.repeat(positive_data, length_neg, axis=0)
    # tile 整体重复
    negetive_tile_data = np.tile(negative_data, (length_pos, 1))


    transfrom_positive_data = positive_repeat_data - negetive_tile_data
    transform_positive_label = np.ones(all_generate_num).reshape(-1, 1)

    transfrom_negetive_data = negetive_tile_data - positive_repeat_data 
    transform_negetive_label = np.zeros(all_generate_num).reshape(-1, 1)

    all_transformed_data = np.vstack( (transfrom_positive_data, transfrom_negetive_data) )
    all_transformed_label = np.vstack( (transform_positive_label, transform_negetive_label) )

    all_data_label = np.hstack( (all_transformed_data, all_transformed_label) )
    np.random.shuffle(all_data_label)

    transformed_data = all_data_label[:, :-1]
    transformed_label = all_data_label[:, -1].reshape(-1, 1)

    return transformed_data, transformed_label


def handleData_minus_not_mirror(positive_data, negative_data, positive_value=1, negative_value=0):
    # 生成非镜像模式数据
    length_pos = positive_data.shape[0]
    length_neg = negative_data.shape[0]
    all_generate_num = length_pos * length_neg

    # 生成 label 数据，保证同一个组合不会既有正样本，又有负样本
    init_transformed_label = np.random.randint(low=0,high=2,size=all_generate_num).reshape(-1, 1)
    positive_index = np.where(init_transformed_label == 1)
    negetive_index = np.where(init_transformed_label == 0)

    # repeat 每一个都连续重复
    positive_repeat_data = np.repeat(positive_data, length_neg, axis=0)
    # tile 整体重复
    negetive_tile_data = np.tile(negative_data, (length_pos, 1))

    
    transfrom_positive_data = positive_repeat_data - negetive_tile_data
    transfrom_positive_data = transfrom_positive_data[positive_index[0]]
    transform_positive_label = np.ones(transfrom_positive_data.shape[0]).reshape(-1, 1)


    transfrom_negetive_data = negetive_tile_data - positive_repeat_data
    transfrom_negetive_data = transfrom_negetive_data[negetive_index[0]]
    transform_negetive_label = np.zeros(transfrom_negetive_data.shape[0]).reshape(-1, 1)

    all_transformed_data = np.vstack( (transfrom_positive_data, transfrom_negetive_data) )
    all_transformed_label = np.vstack( (transform_positive_label, transform_negetive_label) )

    all_data_label = np.hstack( (all_transformed_data, all_transformed_label) )
    np.random.shuffle(all_data_label)

    transformed_data = all_data_label[:, :-1]
    transformed_label = all_data_label[:, -1].reshape(-1, 1)

    return transformed_data, transformed_label






def handleData_extend_mirror(positive_data, negative_data):
    # 生成镜像数据
    length_pos = positive_data.shape[0]
    length_neg = negative_data.shape[0]
    all_generate_num = length_pos * length_neg

    # repeat 每一个都连续重复
    positive_repeat_data = np.repeat(positive_data, length_neg, axis=0)
    # tile 整体重复
    negetive_tile_data = np.tile(negative_data, (length_pos, 1))


    transfrom_positive_data = np.hstack( (positive_repeat_data, negetive_tile_data) )
    transform_positive_label = np.ones(all_generate_num).reshape(-1, 1)

    transfrom_negetive_data = np.hstack( (negetive_tile_data, positive_repeat_data) )
    transform_negetive_label = np.zeros(all_generate_num).reshape(-1, 1)

    all_transformed_data = np.vstack( (transfrom_positive_data, transfrom_negetive_data) )
    all_transformed_label = np.vstack( (transform_positive_label, transform_negetive_label) )

    all_data_label = np.hstack( (all_transformed_data, all_transformed_label) )
    np.random.shuffle(all_data_label)

    transformed_data = all_data_label[:, :-1]
    transformed_label = all_data_label[:, -1].reshape(-1, 1)

    return transformed_data, transformed_label


def handleData_extend_not_mirror(positive_data, negative_data, positive_value=1, negative_value=0):
    # 生成非镜像模式数据
    length_pos = positive_data.shape[0]
    length_neg = negative_data.shape[0]
    all_generate_num = length_pos * length_neg

    # 生成 label 数据，保证同一个组合不会既有正样本，又有负样本
    init_transformed_label = np.random.randint(low=0,high=2,size=all_generate_num).reshape(-1, 1)
    positive_index = np.where(init_transformed_label == 1)
    negetive_index = np.where(init_transformed_label == 0)

    # repeat 每一个都连续重复
    positive_repeat_data = np.repeat(positive_data, length_neg, axis=0)
    # tile 整体重复
    negetive_tile_data = np.tile(negative_data, (length_pos, 1))    
    
    transfrom_positive_data = np.hstack( (positive_repeat_data, negetive_tile_data) )
    transfrom_positive_data = transfrom_positive_data[positive_index[0]]
    transform_positive_label = np.ones(transfrom_positive_data.shape[0]).reshape(-1, 1)


    transfrom_negetive_data = np.hstack( (negetive_tile_data, positive_repeat_data) )
    transfrom_negetive_data = transfrom_negetive_data[negetive_index[0]]
    transform_negetive_label = np.zeros(transfrom_negetive_data.shape[0]).reshape(-1, 1)

    all_transformed_data = np.vstack( (transfrom_positive_data, transfrom_negetive_data) )
    all_transformed_label = np.vstack( (transform_positive_label, transform_negetive_label) )

    all_data_label = np.hstack( (all_transformed_data, all_transformed_label) )
    np.random.shuffle(all_data_label)

    transformed_data = all_data_label[:, :-1]
    transformed_label = all_data_label[:, -1].reshape(-1, 1)

    return transformed_data, transformed_label

def loadTrainData(file_name):
    file_data = np.loadtxt(file_name, delimiter=',')
    label = file_data[:,-1]
    data = np.delete(file_data, -1, axis=1)
    data = data.astype(np.float64)
    label = label.reshape(-1, 1)
    label = label.astype(np.int)
    return data, label


def set_para():
    global dataset_name
    global dataset_index
    global record_index

    argv = sys.argv[1:]
    for each in argv:
        para = each.split('=')
        if para[0] == 'dataset_name':
            dataset_name = para[1]
        if para[0] == 'dataset_index':
            dataset_index = para[1]
        if para[0] == 'record_index':
            record_index = para[1]

# -------------------------------------parameters----------------------------------------
print('draw_original_tsne\n')
dataset_name = 'abalone19'
dataset_index = '1'
record_index = '1'

# ----------------------------------set parameters---------------------------------------
set_para()
train_file_name = './test_{0}/standlization_data/{0}_std_train_{1}.csv'.format(dataset_name, dataset_index)
record_path = './test_{0}/draw_pca_pic/record_{1}/'.format(dataset_name, record_index)

# ----------------------------------start processing-------------------------------------
print('{0}_{1}_draw_original_tsne\n'.format(dataset_name, dataset_index))
print(train_file_name)
print(record_path)
print('----------------------\n\n\n')


# 原始数据 pca
train_data, train_label = loadTrainData(train_file_name)
tsne_train_data = condense_data_tsne(train_data)

pos_tsne_train_data, neg_tsne_train_data = divide_data(tsne_train_data, train_label)

def plot_tsne_fig(pos_data, neg_data, record_path, dataset_name, dataset_index):
    '''
        仅绘画一张图，根据 tsne 绘制
    '''
    # 画图部分，plt 比较麻烦，暂时就重复建图了

    plt.figure()
    plt.scatter(pos_data[:, 0], pos_data[:, 1], s=1, color='r', label='pos')
    plt.scatter(neg_data[:, 0], neg_data[:, 1], s=1, color='b', label='neg')
    # plt.set_title('original_data_tsne')
    # plt.xlabel('test week')
    # plt.ylabel('Fscore')
    plt.legend(loc='upper left')

    record_file_postfix = 'original_data_tsne'
    plt.title(dataset_name + '_' + dataset_index + '_' + record_file_postfix)
    record_file_name = record_path + '{0}_{1}_'.format(dataset_name, dataset_index) + record_file_postfix + '.pdf'

    plt.savefig(record_file_name)

plot_tsne_fig(pos_tsne_train_data, neg_tsne_train_data, record_path, dataset_name, dataset_index )