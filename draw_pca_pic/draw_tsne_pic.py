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
from sklearn.manifold import TSNE 
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





def handleData_extend_ijcai(positive_data, negative_data, positive_value=1, negative_value=0):
    # 生成非镜像模式数据
    length_pos = positive_data.shape[0]
    length_neg = negative_data.shape[0]
    all_generate_num = length_pos * length_neg

    # 生成 label 数据，保证同一个组合不会既有正样本，又有负样本
    # init_transformed_nn_label = np.random.randint(low=0,high=2,size=length_neg*length_neg).reshape(-1, 1)
    # active_index = np.where(init_transformed_nn_label == 1)
    # negetive_index = np.where(init_transformed_label == 0)

    active_index = np.random.choice(length_neg*length_neg, length_neg*length_pos, replace=False)

    # repeat 每一个都连续重复
    positive_repeat_data = np.repeat(positive_data, length_neg, axis=0)
    positive_tile_data = np.tile(positive_data, (length_neg, 1))
    # tile 整体重复
    negetive_repeat_data = np.repeat(negative_data, length_pos, axis=0)
    negetive_tile_data = np.tile(negative_data, (length_pos, 1))
    
    nn_repeat_data = np.repeat(negative_data, length_neg, axis=0)
    nn_tile_data = np.tile(negative_data, (length_neg, 1))
    
    
    transform_nn_data = np.hstack( (nn_repeat_data, nn_tile_data) )
    transform_nn_data = transform_nn_data[active_index]
    transform_nn_label = np.ones(transform_nn_data.shape[0]).reshape(-1, 1) * 1
    

    transform_pn_data = np.hstack( (positive_repeat_data, negetive_tile_data) )
    transform_pn_label = np.ones(transform_pn_data.shape[0]).reshape(-1, 1) * 2

    transform_np_data = np.hstack( (negetive_repeat_data, positive_tile_data) )
    transform_np_label = np.ones(transform_np_data.shape[0]).reshape(-1, 1) * 3

    transform_pp_data = np.hstack( (positive_repeat_data, positive_tile_data) )
    transform_pp_label = np.ones(transform_np_data.shape[0]).reshape(-1, 1) * 4
    
    nn_pn_transform_data = np.vstack((transform_nn_data, transform_pn_data))
    np_pp_transform_data = np.vstack((transform_np_data, transform_pp_data))
    all_transform_data = np.vstack((nn_pn_transform_data, np_pp_transform_data))

    nn_pn_transform_label = np.vstack((transform_nn_label, transform_pn_label))
    np_pp_transform_label = np.vstack((transform_np_label, transform_pp_label))
    all_transform_label = np.vstack((nn_pn_transform_label, np_pp_transform_label))

    # all_data_label = np.hstack( (all_transform_data, all_transform_label) )

    return all_transform_data, all_transform_label








def handleData_minus_ijcai(positive_data, negative_data, positive_value=1, negative_value=0):
    # 生成非镜像模式数据
    length_pos = positive_data.shape[0]
    length_neg = negative_data.shape[0]
    all_generate_num = length_pos * length_neg

    active_index = np.random.choice(length_neg*length_neg, length_neg*length_pos, replace=False)

    # repeat 每一个都连续重复
    positive_repeat_data = np.repeat(positive_data, length_neg, axis=0)
    positive_tile_data = np.tile(positive_data, (length_neg, 1))
    # tile 整体重复
    negetive_repeat_data = np.repeat(negative_data, length_pos, axis=0)
    negetive_tile_data = np.tile(negative_data, (length_pos, 1))
    
    nn_repeat_data = np.repeat(negative_data, length_neg, axis=0)
    nn_tile_data = np.tile(negative_data, (length_neg, 1))
    
    
    transform_nn_data = nn_repeat_data - nn_tile_data
    transform_nn_data = transform_nn_data[active_index]
    transform_nn_label = np.ones(transform_nn_data.shape[0]).reshape(-1, 1) * 1
    

    transform_pn_data = positive_repeat_data - negetive_tile_data
    transform_pn_label = np.ones(transform_pn_data.shape[0]).reshape(-1, 1) * 2

    transform_np_data = negetive_repeat_data - positive_tile_data
    transform_np_label = np.ones(transform_np_data.shape[0]).reshape(-1, 1) * 3

    transform_pp_data = positive_repeat_data - positive_tile_data
    transform_pp_label = np.ones(transform_np_data.shape[0]).reshape(-1, 1) * 4
    
    nn_pn_transform_data = np.vstack((transform_nn_data, transform_pn_data))
    np_pp_transform_data = np.vstack((transform_np_data, transform_pp_data))
    all_transform_data = np.vstack((nn_pn_transform_data, np_pp_transform_data))

    nn_pn_transform_label = np.vstack((transform_nn_label, transform_pn_label))
    np_pp_transform_label = np.vstack((transform_np_label, transform_pp_label))
    all_transform_label = np.vstack((nn_pn_transform_label, np_pp_transform_label))

    # all_data_label = np.hstack( (all_transform_data, all_transform_label) )

    return all_transform_data, all_transform_label










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
train_file_name = './test_{0}/standlization_data/{0}_std_train_{1}.csv'.format(dataset_name, dataset_index)
# test_file_name = './test_{0}/standlization_data/{0}_std_test_{1}.csv'.format(dataset_name, dataset_index)
record_path = './test_{0}/tsne_2_pic/'.format(dataset_name)
record_file_name = record_path + '{0}_tsne_{1}.pdf'.format(dataset_name, dataset_index)
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
tsne_train_data = condense_data_tsne(train_data)
# kernel_pca_train_data = condense_data_kernel_pca(train_data)

positive_data, negative_data = divide_data(train_data, train_label)






# concat not mirror
concat_ijcai_train_data, concat_ijcai_train_label = handleData_extend_ijcai(positive_data, negative_data)
print('data_transform')
tsne_concat_ijcai_train_data = condense_data_tsne(concat_ijcai_train_data)
print('pac or tsne')

# transform_nn_label = np.ones(transform_nn_data.shape[0]).reshape(-1, 1) * 1
# transform_pn_label = np.ones(transform_pn_data.shape[0]).reshape(-1, 1) * 2
# transform_np_label = np.ones(transform_np_data.shape[0]).reshape(-1, 1) * 3
# transform_pp_label = np.ones(transform_np_data.shape[0]).reshape(-1, 1) * 4

nn_index = np.where(concat_ijcai_train_label == 1)
pn_index = np.where(concat_ijcai_train_label == 2)
np_index = np.where(concat_ijcai_train_label == 3)
pp_index = np.where(concat_ijcai_train_label == 4)

# pos_tsne_train_data, neg_tsne_train_data = divide_data(tsne_concat_not_mirror_train_data, concat_not_mirror_train_label)

nn_data = tsne_concat_ijcai_train_data[nn_index[0]]
pn_data = tsne_concat_ijcai_train_data[pn_index[0]]
np_data = tsne_concat_ijcai_train_data[np_index[0]]
pp_data = tsne_concat_ijcai_train_data[pp_index[0]]



plt.figure()

plt.scatter(nn_data[:, 0], nn_data[:, 1], s=1, color='r', label='Neg-Neg')
plt.scatter(pn_data[:, 0], pn_data[:, 1], s=1, color='b', label='pos-Neg')
plt.scatter(np_data[:, 0], np_data[:, 1], s=1, color='g', label='Neg-pos')
plt.scatter(pp_data[:, 0], pp_data[:, 1], s=1, color='m', label='pos-pos')
plt.axis('square')

# plt.set_title('Original data distribution')
# plt.xlabel('test week')
# plt.ylabel('Fscore')
plt.legend(loc='upper left')

record_file_postfix = 'Transformed_data_distribution_ijcai_concat'
# # plt.title(dataset_name + '_' + dataset_index + '_' + record_file_postfix)
record_file_name = record_path + '{0}_{1}_'.format(dataset_name, dataset_index) + record_file_postfix + '.pdf'

plt.savefig(record_file_name)







# concat not mirror
concat_ijcai_train_data, concat_ijcai_train_label = handleData_minus_ijcai(positive_data, negative_data)
print('data_transform')
tsne_concat_ijcai_train_data = condense_data_tsne(concat_ijcai_train_data)
print('pac or tsne')

# transform_nn_label = np.ones(transform_nn_data.shape[0]).reshape(-1, 1) * 1
# transform_pn_label = np.ones(transform_pn_data.shape[0]).reshape(-1, 1) * 2
# transform_np_label = np.ones(transform_np_data.shape[0]).reshape(-1, 1) * 3
# transform_pp_label = np.ones(transform_np_data.shape[0]).reshape(-1, 1) * 4

nn_index = np.where(concat_ijcai_train_label == 1)
pn_index = np.where(concat_ijcai_train_label == 2)
np_index = np.where(concat_ijcai_train_label == 3)
pp_index = np.where(concat_ijcai_train_label == 4)

# pos_tsne_train_data, neg_tsne_train_data = divide_data(tsne_concat_not_mirror_train_data, concat_not_mirror_train_label)

nn_data = tsne_concat_ijcai_train_data[nn_index[0]]
pn_data = tsne_concat_ijcai_train_data[pn_index[0]]
np_data = tsne_concat_ijcai_train_data[np_index[0]]
pp_data = tsne_concat_ijcai_train_data[pp_index[0]]



plt.figure()

plt.scatter(nn_data[:, 0], nn_data[:, 1], s=1, color='r', label='Neg-Neg')
plt.scatter(pn_data[:, 0], pn_data[:, 1], s=1, color='b', label='pos-Neg')
plt.scatter(np_data[:, 0], np_data[:, 1], s=1, color='g', label='Neg-pos')
plt.scatter(pp_data[:, 0], pp_data[:, 1], s=1, color='m', label='pos-pos')
plt.axis('square')

# plt.set_title('Original data distribution')
# plt.xlabel('test week')
# plt.ylabel('Fscore')
plt.legend(loc='upper left')

record_file_postfix = 'Transformed_data_distribution_ijcai_minus'
# # plt.title(dataset_name + '_' + dataset_index + '_' + record_file_postfix)
record_file_name = record_path + '{0}_{1}_'.format(dataset_name, dataset_index) + record_file_postfix + '.pdf'

plt.savefig(record_file_name)


























# original_pos_data, original_neg_data = divide_data(tsne_train_data, train_label)

# plt.figure()

# plt.scatter(original_pos_data[:, 0], original_pos_data[:, 1], s=1, color='r', label='pos')
# plt.scatter(original_neg_data[:, 0], original_neg_data[:, 1], s=1, color='b', label='neg')
# plt.axis('square')

# # plt.set_title('Original data distribution')
# # plt.xlabel('test week')
# # plt.ylabel('Fscore')
# plt.legend(loc='upper left')

# record_file_postfix = 'original_data_tsne'
# # plt.title(dataset_name + '_' + dataset_index + '_' + record_file_postfix)
# record_file_name = record_path + '{0}_{1}_'.format(dataset_name, dataset_index) + record_file_postfix + '.pdf'

# plt.savefig(record_file_name)






# concat mirror
# concat_mirror_train_data, concat_mirror_train_label = handleData_extend_mirror(positive_data, negative_data)
# pca_concat_mirror_train_data = condense_data_tsne(concat_mirror_train_data)
# # kernel_pca_concat_mirror_train_data = condense_data_kernel_pca(concat_mirror_train_data)


# # concat not mirror
# concat_not_mirror_train_data, concat_not_mirror_train_label = handleData_extend_not_mirror(positive_data, negative_data)
# print('data_transform')
# tsne_concat_not_mirror_train_data = condense_data_tsne(concat_not_mirror_train_data)
# print('pac or tsne')
# pos_tsne_train_data, neg_tsne_train_data = divide_data(tsne_concat_not_mirror_train_data, concat_not_mirror_train_label)



# plt.figure()

# plt.scatter(pos_tsne_train_data[:, 0], pos_tsne_train_data[:, 1], s=1, color='r', label='pos')
# plt.scatter(neg_tsne_train_data[:, 0], neg_tsne_train_data[:, 1], s=1, color='b', label='neg')
# plt.axis('square')

# # plt.set_title('Original data distribution')
# # plt.xlabel('test week')
# # plt.ylabel('Fscore')
# plt.legend(loc='upper left')

# record_file_postfix = 'Transformed_data_distribution_concat_not_mirror'
# # # plt.title(dataset_name + '_' + dataset_index + '_' + record_file_postfix)
# record_file_name = record_path + '{0}_{1}_'.format(dataset_name, dataset_index) + record_file_postfix + '.pdf'

# plt.savefig(record_file_name)




# # concat mirror
# concat_mirror_train_data, concat_mirror_train_label = handleData_extend_mirror(positive_data, negative_data)
# print('data_transform')
# tsne_concat_mirror_train_data = condense_data_tsne(concat_mirror_train_data)
# print('pac or tsne')
# pos_tsne_train_data, neg_tsne_train_data = divide_data(tsne_concat_mirror_train_data, concat_mirror_train_label)



# plt.figure()

# plt.scatter(pos_tsne_train_data[:, 0], pos_tsne_train_data[:, 1], s=1, color='r', label='pos')
# plt.scatter(neg_tsne_train_data[:, 0], neg_tsne_train_data[:, 1], s=1, color='b', label='neg')
# plt.axis('square')

# # plt.set_title('Original data distribution')
# # plt.xlabel('test week')
# # plt.ylabel('Fscore')
# plt.legend(loc='upper left')

# record_file_postfix = 'Transformed_data_distribution_concat_mirror'
# # plt.title(dataset_name + '_' + dataset_index + '_' + record_file_postfix)
# record_file_name = record_path + '{0}_{1}_'.format(dataset_name, dataset_index) + record_file_postfix + '.pdf'

# plt.savefig(record_file_name)








# # minus mirror
# minus_mirror_train_data, minus_mirror_train_label = handleData_minus_mirror(positive_data, negative_data)
# print('data_transform')
# tsne_minus_mirror_train_data = condense_data_tsne(minus_mirror_train_data)
# print('pac or tsne')
# pos_tsne_train_data, neg_tsne_train_data = divide_data(tsne_minus_mirror_train_data, minus_mirror_train_label)



# plt.figure()

# plt.scatter(pos_tsne_train_data[:, 0], pos_tsne_train_data[:, 1], s=1, color='r', label='pos')
# plt.scatter(neg_tsne_train_data[:, 0], neg_tsne_train_data[:, 1], s=1, color='b', label='neg')
# plt.axis('square')

# # plt.set_title('Original data distribution')
# # plt.xlabel('test week')
# # plt.ylabel('Fscore')
# plt.legend(loc='upper left')

# record_file_postfix = 'Transformed_data_distribution_minus_mirror'
# # plt.title(dataset_name + '_' + dataset_index + '_' + record_file_postfix)
# record_file_name = record_path + '{0}_{1}_'.format(dataset_name, dataset_index) + record_file_postfix + '.pdf'

# plt.savefig(record_file_name)










# # minus not mirror
# minus_not_mirror_train_data, minus_not_mirror_train_label = handleData_minus_not_mirror(positive_data, negative_data)
# print('data_transform')
# tsne_minus_not_mirror_train_data = condense_data_tsne(minus_not_mirror_train_data)
# print('pac or tsne')
# pos_tsne_train_data, neg_tsne_train_data = divide_data(tsne_minus_not_mirror_train_data, minus_not_mirror_train_label)





# plt.figure()

# plt.scatter(pos_tsne_train_data[:, 0], pos_tsne_train_data[:, 1], s=1, color='r', label='pos')
# plt.scatter(neg_tsne_train_data[:, 0], neg_tsne_train_data[:, 1], s=1, color='b', label='neg')
# plt.axis('square')

# # plt.set_title('Original data distribution')
# # plt.xlabel('test week')
# # plt.ylabel('Fscore')
# plt.legend(loc='upper left')

# record_file_postfix = 'Transformed_data_distribution_minus_not_mirror'
# # plt.title(dataset_name + '_' + dataset_index + '_' + record_file_postfix)
# record_file_name = record_path + '{0}_{1}_'.format(dataset_name, dataset_index) + record_file_postfix + '.pdf'

# plt.savefig(record_file_name)


































# # 画图部分，plt 比较麻烦，暂时就重复建图了
# fig, [ax1, ax2] = plt.subplots(1,2)
# # ax2 = ax1.twinx() twinx 用于新增个图

# # 原始数据 pca 及 kernel pca
# original_pos_data, original_neg_data = divide_data(tsne_train_data, train_label)
# # concat_not_mirror_pos_data, concat_not_mirror_neg_data = divide_data(kernel_pca_data, label)

# ax1.scatter(original_pos_data[:, 0], original_pos_data[:, 1], s=1, color='r', label='pos')
# ax1.scatter(original_neg_data[:, 0], original_neg_data[:, 1], s=1, color='b', label='neg')
# ax1.set_aspect(1)
# ax1.set_title('Original data distribution')
# ax1.legend(loc=0)

# ax2.scatter(pos_tsne_train_data[:, 0], pos_tsne_train_data[:, 1], s=1, color='r', label='pos')
# ax2.scatter(neg_tsne_train_data[:, 0], neg_tsne_train_data[:, 1], s=1, color='b', label='neg')
# ax1.set_aspect(1)
# ax2.set_title('Transformed data distribution')
# ax2.legend(loc=0)

# record_file_postfix = 'original_transformed_distribution'
# # fig.suptitle(dataset_name + '_' + dataset_index + '_' + record_file_postfix)
# record_file_name = record_path + '{0}_{1}_'.format(dataset_name, dataset_index) + record_file_postfix + '.pdf'

# plt.savefig(record_file_name)




    

# plot_two_pca_fig(pca_train_data, kernel_pca_train_data, train_label, record_path, dataset_name, dataset_index )









# # concat not mirror
# concat_not_mirror_train_data, concat_not_mirror_train_label = handleData_extend_not_mirror(positive_data, negative_data)
# pca_concat_not_mirror_train_data = condense_data_pca(concat_not_mirror_train_data)
# kernel_pca_concat_not_mirror_train_data = condense_data_kernel_pca(concat_not_mirror_train_data)

# # minus mirror
# minus_mirror_train_data, minus_mirror_train_label = handleData_minus_mirror(positive_data, negative_data)
# pca_minus_mirror_train_data = condense_data_pca(minus_mirror_train_data)
# kernel_pca_minus_mirror_train_data = condense_data_kernel_pca(minus_mirror_train_data)

# # minus not mirror
# minus_not_mirror_train_data, minus_not_mirror_train_label = handleData_minus_not_mirror(positive_data, negative_data)
# pca_minus_not_mirror_train_data = condense_data_pca(minus_not_mirror_train_data)
# kernel_pca_minus_not_mirror_train_data = condense_data_kernel_pca(minus_not_mirror_train_data)


# 画图部分，plt 比较麻烦，暂时就重复建图了
# fig, axs = plt.subplots(5,2)
# ax2 = ax1.twinx() twinx 用于新增个图

# # 原始数据 pca 及 kernel pca
# ax01, ax02 = axs[0]
# pca_pos_train_data, pca_neg_train_data = divide_data(pca_train_data, train_label)
# kernel_pca_pos_train_data, kernel_pca_neg_train_data = divide_data(kernel_pca_train_data, train_label)

# ax01.scatter(pca_pos_train_data[:, 0], pca_pos_train_data[:, 1], s=1, color='r', label='pos')
# ax01.scatter(pca_neg_train_data[:, 0], pca_neg_train_data[:, 1], s=1, color='b', label='neg')
# ax01.set_title('original pca')
# ax01.legend(loc=0)

# ax02.scatter(kernel_pca_pos_train_data[:, 0], kernel_pca_pos_train_data[:, 1], s=1, color='r', label='pos')
# ax02.scatter(kernel_pca_neg_train_data[:, 0], kernel_pca_neg_train_data[:, 1], s=1, color='b', label='neg')
# ax02.set_title('original kernel pca')
# ax02.legend(loc=0)



# # concat mirror 数据 pca 及 kernel pca
# ax11, ax12 = axs[1]
# pca_pos_concat_mirror_data, pca_neg_concat_mirror_data = divide_data(pca_concat_mirror_train_data, concat_mirror_train_label)
# kernel_pca_pos_concat_mirror_data, kernel_pca_neg_concat_mirror_data = divide_data(kernel_pca_concat_mirror_train_data, concat_mirror_train_label)

# ax11.scatter(pca_pos_concat_mirror_data[:, 0], pca_pos_concat_mirror_data[:, 1], s=1, color='r', label='pos')
# ax11.scatter(pca_neg_concat_mirror_data[:, 0], pca_neg_concat_mirror_data[:, 1], s=1, color='b', label='neg')
# ax11.set_title('concat mirror pca')
# ax11.legend(loc=0)

# ax12.scatter(kernel_pca_pos_concat_mirror_data[:, 0], kernel_pca_pos_concat_mirror_data[:, 1], s=1, color='r', label='pos')
# ax12.scatter(kernel_pca_neg_concat_mirror_data[:, 0], kernel_pca_neg_concat_mirror_data[:, 1], s=1, color='b', label='neg')
# ax12.set_title('concat mirror kernel pca')
# ax12.legend(loc=0)




# # concat not mirror 数据 pca 及 kernel pca
# ax21, ax22 = axs[2]
# pca_pos_concat_not_mirror_data, pca_neg_concat_not_mirror_data = divide_data(pca_concat_not_mirror_train_data, concat_not_mirror_train_label)
# kernel_pca_pos_concat_not_mirror_data, kernel_pca_neg_concat_not_mirror_data = divide_data(kernel_pca_concat_not_mirror_train_data, concat_not_mirror_train_label)

# ax21.scatter(pca_pos_concat_not_mirror_data[:, 0], pca_pos_concat_not_mirror_data[:, 1], s=1, color='r', label='pos')
# ax21.scatter(pca_neg_concat_not_mirror_data[:, 0], pca_neg_concat_not_mirror_data[:, 1], s=1, color='b', label='neg')
# ax21.set_title('concat not mirror pca')
# ax21.legend(loc=0)

# ax22.scatter(kernel_pca_pos_concat_not_mirror_data[:, 0], kernel_pca_pos_concat_not_mirror_data[:, 1], color='r', label='pos')
# ax22.scatter(kernel_pca_neg_concat_not_mirror_data[:, 0], kernel_pca_neg_concat_not_mirror_data[:, 1], color='b', label='neg')
# ax22.set_title('concat not mirror kernel pca')
# ax22.legend(loc=0)



# # minus mirror 数据 pca 及 kernel pca
# ax31, ax32 = axs[3]
# pca_pos_minus_mirror_data, pca_neg_minus_mirror_data = divide_data(pca_minus_mirror_train_data, minus_mirror_train_label)
# kernel_pca_pos_minus_mirror_data, kernel_pca_neg_minus_mirror_data = divide_data(kernel_pca_minus_mirror_train_data, minus_mirror_train_label)

# ax31.scatter(pca_pos_minus_mirror_data[:, 0], pca_pos_minus_mirror_data[:, 1], s=1, color='r', label='pos')
# ax31.scatter(pca_neg_minus_mirror_data[:, 0], pca_neg_minus_mirror_data[:, 1], s=1, color='b', label='neg')
# ax31.set_title('minus mirror pca')
# ax31.legend(loc=0)

# ax32.scatter(kernel_pca_pos_minus_mirror_data[:, 0], kernel_pca_pos_minus_mirror_data[:, 1], s=1, color='r', label='pos')
# ax32.scatter(kernel_pca_neg_minus_mirror_data[:, 0], kernel_pca_neg_minus_mirror_data[:, 1], s=1, color='b', label='neg')
# ax32.set_title('minus mirror kernel pca')
# ax32.legend(loc=0)




# # minus not mirror 数据 pca 及 kernel pca
# ax41, ax42 = axs[4]
# pca_pos_minus_not_mirror_data, pca_neg_minus_not_mirror_data = divide_data(pca_minus_not_mirror_train_data, minus_not_mirror_train_label)
# kernel_pca_pos_minus_not_mirror_data, kernel_pca_neg_minus_not_mirror_data = divide_data(kernel_pca_minus_not_mirror_train_data, minus_not_mirror_train_label)

# ax41.scatter(pca_pos_minus_not_mirror_data[:, 0], pca_pos_minus_not_mirror_data[:, 1], s=1, color='r', label='pos')
# ax41.scatter(pca_neg_minus_not_mirror_data[:, 0], pca_neg_minus_not_mirror_data[:, 1], s=1, color='b', label='neg')
# ax41.set_title('minus not mirror pca')
# ax41.legend(loc=0)

# ax42.scatter(kernel_pca_pos_minus_not_mirror_data[:, 0], kernel_pca_pos_minus_not_mirror_data[:, 1], s=1, color='r', label='pos')
# ax42.scatter(kernel_pca_neg_minus_not_mirror_data[:, 0], kernel_pca_neg_minus_not_mirror_data[:, 1], s=1, color='b', label='neg')
# ax42.set_title('minus not mirror kernel pca')
# ax42.legend(loc=0)




# plt.title(dataset_name + '_' + dataset_index)
# plt.savefig(record_file_name)



