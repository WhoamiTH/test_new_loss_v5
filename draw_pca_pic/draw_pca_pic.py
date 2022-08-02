# -*- coding: utf-8 -*-

'''
    将所有数据都画在一张图里，但是会有超内存的问题，所以需要分开运算，分开画，全放在一个文件里不行
'''

import sys
# sys.path.append('..')
# import sklearn.svm as sksvm
# import sklearn.linear_model as sklin
# import sklearn.tree as sktree
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
# import joblib
import time
import sklearn.preprocessing as skpre
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt


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
















# def handleData_minus_mirror(positive_data, negative_data):
#     # 生成镜像数据
#     length_pos = positive_data.shape[0]
#     length_neg = negative_data.shape[0]
#     all_generate_num = length_pos * length_neg

#     # repeat 每一个都连续重复
#     positive_repeat_data = np.repeat(positive_data, length_neg, axis=0)
#     # tile 整体重复
#     negetive_tile_data = np.tile(negative_data, (length_pos, 1))


#     transfrom_positive_data = positive_repeat_data - negetive_tile_data
#     transform_positive_label = np.ones(all_generate_num).reshape(-1, 1)

#     transfrom_negetive_data = negetive_tile_data - positive_repeat_data 
#     transform_negetive_label = np.zeros(all_generate_num).reshape(-1, 1)

#     all_transformed_data = np.vstack( (transfrom_positive_data, transfrom_negetive_data) )
#     all_transformed_label = np.vstack( (transform_positive_label, transform_negetive_label) )

#     all_data_label = np.hstack( (all_transformed_data, all_transformed_label) )
#     np.random.shuffle(all_data_label)

#     transformed_data = all_data_label[:, :-1]
#     transformed_label = all_data_label[:, -1].reshape(-1, 1)

#     return transformed_data, transformed_label


# def handleData_minus_not_mirror(positive_data, negative_data, positive_value=1, negative_value=0):
#     # 生成非镜像模式数据
#     length_pos = positive_data.shape[0]
#     length_neg = negative_data.shape[0]
#     all_generate_num = length_pos * length_neg

#     # 生成 label 数据，保证同一个组合不会既有正样本，又有负样本
#     init_transformed_label = np.random.randint(low=0,high=2,size=all_generate_num).reshape(-1, 1)
#     positive_index = np.where(init_transformed_label == 1)
#     negetive_index = np.where(init_transformed_label == 0)

#     # repeat 每一个都连续重复
#     positive_repeat_data = np.repeat(positive_data, length_neg, axis=0)
#     # tile 整体重复
#     negetive_tile_data = np.tile(negative_data, (length_pos, 1))

    
#     transfrom_positive_data = positive_repeat_data - negetive_tile_data
#     transfrom_positive_data = transfrom_positive_data[positive_index[0]]
#     transform_positive_label = np.ones(transfrom_positive_data.shape[0]).reshape(-1, 1)


#     transfrom_negetive_data = negetive_tile_data - positive_repeat_data
#     transfrom_negetive_data = transfrom_negetive_data[negetive_index[0]]
#     transform_negetive_label = np.zeros(transfrom_negetive_data.shape[0]).reshape(-1, 1)

#     all_transformed_data = np.vstack( (transfrom_positive_data, transfrom_negetive_data) )
#     all_transformed_label = np.vstack( (transform_positive_label, transform_negetive_label) )

#     all_data_label = np.hstack( (all_transformed_data, all_transformed_label) )
#     np.random.shuffle(all_data_label)

#     transformed_data = all_data_label[:, :-1]
#     transformed_label = all_data_label[:, -1].reshape(-1, 1)

#     return transformed_data, transformed_label






# def handleData_extend_mirror(positive_data, negative_data):
#     # 生成镜像数据
#     length_pos = positive_data.shape[0]
#     length_neg = negative_data.shape[0]
#     all_generate_num = length_pos * length_neg

#     # repeat 每一个都连续重复
#     positive_repeat_data = np.repeat(positive_data, length_neg, axis=0)
#     # tile 整体重复
#     negetive_tile_data = np.tile(negative_data, (length_pos, 1))


#     transfrom_positive_data = np.hstack( (positive_repeat_data, negetive_tile_data) )
#     transform_positive_label = np.ones(all_generate_num).reshape(-1, 1)

#     transfrom_negetive_data = np.hstack( (negetive_tile_data, positive_repeat_data) )
#     transform_negetive_label = np.zeros(all_generate_num).reshape(-1, 1)

#     all_transformed_data = np.vstack( (transfrom_positive_data, transfrom_negetive_data) )
#     all_transformed_label = np.vstack( (transform_positive_label, transform_negetive_label) )

#     all_data_label = np.hstack( (all_transformed_data, all_transformed_label) )
#     np.random.shuffle(all_data_label)

#     transformed_data = all_data_label[:, :-1]
#     transformed_label = all_data_label[:, -1].reshape(-1, 1)

#     return transformed_data, transformed_label


# def handleData_extend_not_mirror(positive_data, negative_data, positive_value=1, negative_value=0):
#     # 生成非镜像模式数据
#     length_pos = positive_data.shape[0]
#     length_neg = negative_data.shape[0]
#     all_generate_num = length_pos * length_neg

#     # 生成 label 数据，保证同一个组合不会既有正样本，又有负样本
#     init_transformed_label = np.random.randint(low=0,high=2,size=all_generate_num).reshape(-1, 1)
#     positive_index = np.where(init_transformed_label == 1)
#     negetive_index = np.where(init_transformed_label == 0)

#     # repeat 每一个都连续重复
#     positive_repeat_data = np.repeat(positive_data, length_neg, axis=0)
#     # tile 整体重复
#     negetive_tile_data = np.tile(negative_data, (length_pos, 1))    
    
#     transfrom_positive_data = np.hstack( (positive_repeat_data, negetive_tile_data) )
#     transfrom_positive_data = transfrom_positive_data[positive_index[0]]
#     transform_positive_label = np.ones(transfrom_positive_data.shape[0]).reshape(-1, 1)


#     transfrom_negetive_data = np.hstack( (negetive_tile_data, positive_repeat_data) )
#     transfrom_negetive_data = transfrom_negetive_data[negetive_index[0]]
#     transform_negetive_label = np.zeros(transfrom_negetive_data.shape[0]).reshape(-1, 1)

#     all_transformed_data = np.vstack( (transfrom_positive_data, transfrom_negetive_data) )
#     all_transformed_label = np.vstack( (transform_positive_label, transform_negetive_label) )

#     all_data_label = np.hstack( (all_transformed_data, all_transformed_label) )
#     np.random.shuffle(all_data_label)

#     transformed_data = all_data_label[:, :-1]
#     transformed_label = all_data_label[:, -1].reshape(-1, 1)

#     return transformed_data, transformed_label

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
dataset_name = 'abalone19'
dataset_index = '1'
record_index = '1'

# ----------------------------------set parameters---------------------------------------
set_para()
train_file_name = './test_{0}/standlization_data/{0}_std_train_{1}.csv'.format(dataset_name, dataset_index)
# test_file_name = './test_{0}/standlization_data/{0}_std_test_{1}.csv'.format(dataset_name, dataset_index)
record_path = './test_{0}/draw_pca_pic/record_{1}/'.format(dataset_name, record_index)
record_file_name = record_path + '{0}_pca_{1}.pdf'.format(dataset_name, dataset_index)
# scaler_name = record_path + 'scaler_' + dataset_index + '.m'
# record_train_file_name = record_path + '{0}_std_train_{1}.csv'.format(dataset_name, dataset_index)
# record_test_file_name = record_path + '{0}_std_test_{1}.csv'.format(dataset_name, dataset_index)

# ----------------------------------start processing-------------------------------------
print(train_file_name)
# print(test_file_name)
print(record_path)
# print(scaler_name)
print(record_file_name)
print('----------------------\n\n\n')


# 原始数据 pca
train_data, train_label = loadTrainData(train_file_name)
pca_train_data = condense_data_pca(train_data)
kernel_pca_train_data = condense_data_kernel_pca(train_data)



positive_data, negative_data = divide_data(train_data, train_label)

# concat mirror
concat_mirror_train_data, concat_mirror_train_label = handleData_extend_mirror(positive_data, negative_data)
pca_concat_mirror_train_data = condense_data_pca(concat_mirror_train_data)
kernel_pca_concat_mirror_train_data = condense_data_kernel_pca(concat_mirror_train_data)


# concat not mirror
concat_not_mirror_train_data, concat_not_mirror_train_label = handleData_extend_not_mirror(positive_data, negative_data)
pca_concat_not_mirror_train_data = condense_data_pca(concat_not_mirror_train_data)
kernel_pca_concat_not_mirror_train_data = condense_data_kernel_pca(concat_not_mirror_train_data)

# minus mirror
minus_mirror_train_data, minus_mirror_train_label = handleData_minus_mirror(positive_data, negative_data)
pca_minus_mirror_train_data = condense_data_pca(minus_mirror_train_data)
kernel_pca_minus_mirror_train_data = condense_data_kernel_pca(minus_mirror_train_data)

# minus not mirror
minus_not_mirror_train_data, minus_not_mirror_train_label = handleData_minus_not_mirror(positive_data, negative_data)
pca_minus_not_mirror_train_data = condense_data_pca(minus_not_mirror_train_data)
kernel_pca_minus_not_mirror_train_data = condense_data_kernel_pca(minus_not_mirror_train_data)


# 画图部分，plt 比较麻烦，暂时就重复建图了
fig, axs = plt.subplots(5,2)
# ax2 = ax1.twinx() twinx 用于新增个图

# 原始数据 pca 及 kernel pca
ax01, ax02 = axs[0]
pca_pos_train_data, pca_neg_train_data = divide_data(pca_train_data, train_label)
kernel_pca_pos_train_data, kernel_pca_neg_train_data = divide_data(kernel_pca_train_data, train_label)

ax01.scatter(pca_pos_train_data[:, 0], pca_pos_train_data[:, 1], s=1, color='r', label='pos')
ax01.scatter(pca_neg_train_data[:, 0], pca_neg_train_data[:, 1], s=1, color='b', label='neg')
ax01.set_title('original pca')
ax01.legend(loc=0)

ax02.scatter(kernel_pca_pos_train_data[:, 0], kernel_pca_pos_train_data[:, 1], s=1, color='r', label='pos')
ax02.scatter(kernel_pca_neg_train_data[:, 0], kernel_pca_neg_train_data[:, 1], s=1, color='b', label='neg')
ax02.set_title('original kernel pca')
ax02.legend(loc=0)



# concat mirror 数据 pca 及 kernel pca
ax11, ax12 = axs[1]
pca_pos_concat_mirror_data, pca_neg_concat_mirror_data = divide_data(pca_concat_mirror_train_data, concat_mirror_train_label)
kernel_pca_pos_concat_mirror_data, kernel_pca_neg_concat_mirror_data = divide_data(kernel_pca_concat_mirror_train_data, concat_mirror_train_label)

ax11.scatter(pca_pos_concat_mirror_data[:, 0], pca_pos_concat_mirror_data[:, 1], s=1, color='r', label='pos')
ax11.scatter(pca_neg_concat_mirror_data[:, 0], pca_neg_concat_mirror_data[:, 1], s=1, color='b', label='neg')
ax11.set_title('concat mirror pca')
ax11.legend(loc=0)

ax12.scatter(kernel_pca_pos_concat_mirror_data[:, 0], kernel_pca_pos_concat_mirror_data[:, 1], s=1, color='r', label='pos')
ax12.scatter(kernel_pca_neg_concat_mirror_data[:, 0], kernel_pca_neg_concat_mirror_data[:, 1], s=1, color='b', label='neg')
ax12.set_title('concat mirror kernel pca')
ax12.legend(loc=0)




# concat not mirror 数据 pca 及 kernel pca
ax21, ax22 = axs[2]
pca_pos_concat_not_mirror_data, pca_neg_concat_not_mirror_data = divide_data(pca_concat_not_mirror_train_data, concat_not_mirror_train_label)
kernel_pca_pos_concat_not_mirror_data, kernel_pca_neg_concat_not_mirror_data = divide_data(kernel_pca_concat_not_mirror_train_data, concat_not_mirror_train_label)

ax21.scatter(pca_pos_concat_not_mirror_data[:, 0], pca_pos_concat_not_mirror_data[:, 1], s=1, color='r', label='pos')
ax21.scatter(pca_neg_concat_not_mirror_data[:, 0], pca_neg_concat_not_mirror_data[:, 1], s=1, color='b', label='neg')
ax21.set_title('concat not mirror pca')
ax21.legend(loc=0)

ax22.scatter(kernel_pca_pos_concat_not_mirror_data[:, 0], kernel_pca_pos_concat_not_mirror_data[:, 1], s=1, color='r', label='pos')
ax22.scatter(kernel_pca_neg_concat_not_mirror_data[:, 0], kernel_pca_neg_concat_not_mirror_data[:, 1], s=1, color='b', label='neg')
ax22.set_title('concat not mirror kernel pca')
ax22.legend(loc=0)



# minus mirror 数据 pca 及 kernel pca
ax31, ax32 = axs[3]
pca_pos_minus_mirror_data, pca_neg_minus_mirror_data = divide_data(pca_minus_mirror_train_data, minus_mirror_train_label)
kernel_pca_pos_minus_mirror_data, kernel_pca_neg_minus_mirror_data = divide_data(kernel_pca_minus_mirror_train_data, minus_mirror_train_label)

ax31.scatter(pca_pos_minus_mirror_data[:, 0], pca_pos_minus_mirror_data[:, 1], s=1, color='r', label='pos')
ax31.scatter(pca_neg_minus_mirror_data[:, 0], pca_neg_minus_mirror_data[:, 1], s=1, color='b', label='neg')
ax31.set_title('minus mirror pca')
ax31.legend(loc=0)

ax32.scatter(kernel_pca_pos_minus_mirror_data[:, 0], kernel_pca_pos_minus_mirror_data[:, 1], s=1, color='r', label='pos')
ax32.scatter(kernel_pca_neg_minus_mirror_data[:, 0], kernel_pca_neg_minus_mirror_data[:, 1], s=1, color='b', label='neg')
ax32.set_title('minus mirror kernel pca')
ax32.legend(loc=0)




# minus not mirror 数据 pca 及 kernel pca
ax41, ax42 = axs[4]
pca_pos_minus_not_mirror_data, pca_neg_minus_not_mirror_data = divide_data(pca_minus_not_mirror_train_data, minus_not_mirror_train_label)
kernel_pca_pos_minus_not_mirror_data, kernel_pca_neg_minus_not_mirror_data = divide_data(kernel_pca_minus_not_mirror_train_data, minus_not_mirror_train_label)

ax41.scatter(pca_pos_minus_not_mirror_data[:, 0], pca_pos_minus_not_mirror_data[:, 1], s=1, color='r', label='pos')
ax41.scatter(pca_neg_minus_not_mirror_data[:, 0], pca_neg_minus_not_mirror_data[:, 1], s=1, color='b', label='neg')
ax41.set_title('minus not mirror pca')
ax41.legend(loc=0)

ax42.scatter(kernel_pca_pos_minus_not_mirror_data[:, 0], kernel_pca_pos_minus_not_mirror_data[:, 1], s=1, color='r', label='pos')
ax42.scatter(kernel_pca_neg_minus_not_mirror_data[:, 0], kernel_pca_neg_minus_not_mirror_data[:, 1], s=1, color='b', label='neg')
ax42.set_title('minus not mirror kernel pca')
ax42.legend(loc=0)




plt.title(dataset_name + '_' + dataset_index)
plt.savefig(record_file_name)



