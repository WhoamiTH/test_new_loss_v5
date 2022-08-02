# train processing head
import sys

from sklearn.externals import joblib
from time import clock

import pandas as pd
import numpy as np
import sklearn.metrics as skmet
from sklearn.metrics import accuracy_score
from pytorchtools import EarlyStopping
import torch
from torch import nn
from torch.nn import init
import time


import os

def divide_data(Data, Label):
    positive_index = np.where(Label == 1)
    negative_index = np.where(Label == 0)

    positive = Data[positive_index[0]]
    negative = Data[negative_index[0]]
    return positive, negative

def generate_valid_data(data, label, size=0.05):
    # 按照比例划分训练集 和 validation 集合
    positive_data, negative_data = divide_data(data, label)
    positive_length = positive_data.shape[0]
    negative_length = negative_data.shape[0]
    positive_label = np.ones(positive_length).reshape(-1, 1)
    negative_label = np.zeros(negative_length).reshape(-1, 1)

    positive_data_label = np.hstack((positive_data, positive_label))
    negative_data_label = np.hstack((negative_data, negative_label))

    np.random.shuffle(positive_data_label)
    np.random.shuffle(negative_data_label)

    positive_valid_length = max(int(positive_length * size), 1)
    negative_valid_length = max(int(negative_length * size), 1)
    
    valid_pos_data_label = positive_data_label[:positive_valid_length, :]
    train_pos_data_label = positive_data_label[positive_valid_length:, :]

    valid_neg_data_label = negative_data_label[:negative_valid_length, :]
    train_neg_data_label = negative_data_label[negative_valid_length:, :]

    valid_data_label = np.vstack((valid_pos_data_label, valid_neg_data_label))
    train_data_label = np.vstack((train_pos_data_label, train_neg_data_label))

    np.random.shuffle(valid_data_label)
    np.random.shuffle(train_data_label)

    valid_data = valid_data_label[:, :-1]
    valid_label = valid_data_label[:, -1].reshape(-1, 1)

    train_data = train_data_label[:, :-1]
    train_label = train_data_label[:, -1].reshape(-1, 1)

    return valid_data, valid_label, train_data, train_label

# def generate_transformed_batch_data(sample_method, positive_data, negative_data, batch_size):
#     positive_length = positive_data.shape[0]
#     negative_length = negative_data.shape[0]

#     if sample_method == 'balance':
#         times = 1
#     else:
#         times = negative_length / positive_length
#         times = int(times)

#     current_pos_batch_size = min(positive_length, batch_size)
#     current_neg_batch_size = min(negative_length, times*current_pos_batch_size)

#     pos_sample_index = np.random.choice(positive_length, current_pos_batch_size, replace=False)
#     neg_sample_index = np.random.choice(negative_length, current_neg_batch_size, replace=False)

#     sampled_positive_data = positive_data[pos_sample_index]
#     sampled_negative_data = negative_data[neg_sample_index]
#     return sampled_positive_data, sampled_negative_data


def generate_normal_batch_data(data, label, batch_size):
    data_length = data.shape[0]
    batch_size = min(batch_size, data_length)
    
    data_index = np.random.choice(data_length, batch_size, replace=False)

    train_data = data[data_index]
    train_label = label[data_index]
    
    train_label = train_label.reshape(-1, 1)
    return train_data, train_label

# def generate_batch_data( sample_method, data, label, pos_data, neg_data, batch_size):
    # if sample_method == 'normal':
    # train_data_pos_data, train_label_neg_data = generate_normal_batch_data(data, label, batch_size)
    # else:
    # train_data_pos_data, train_label_neg_data = generate_transformed_batch_data(sample_method, pos_data, neg_data, batch_size)
    # return train_data_pos_data, train_label_neg_data




# def handleData_minus_mirror(positive_repeat_data, negetive_tile_data):
#     all_generate_num = positive_repeat_data.shape[0]
#     transfrom_positive_data = positive_repeat_data - negetive_tile_data
#     transform_positive_label = np.ones(all_generate_num).reshape(-1, 1)

#     transfrom_negetive_data = negetive_tile_data - positive_repeat_data 
#     transform_negetive_label = np.zeros(all_generate_num).reshape(-1, 1)

#     return transfrom_positive_data, transform_positive_label, transfrom_negetive_data, transform_negetive_label


# def handleData_minus_not_mirror(positive_repeat_data, negetive_tile_data):
#     # 生成 label 数据，保证同一个组合不会既有正样本，又有负样本
#     all_generate_num = positive_repeat_data.shape[0]
#     init_transformed_label = np.random.randint(low=0,high=2,size=all_generate_num).reshape(-1, 1)
#     positive_index = np.where(init_transformed_label == 1)
#     negetive_index = np.where(init_transformed_label == 0)

#     transfrom_positive_data = positive_repeat_data - negetive_tile_data
#     transfrom_positive_data = transfrom_positive_data[positive_index[0]]
#     transform_positive_label = np.ones(transfrom_positive_data.shape[0]).reshape(-1, 1)


#     transfrom_negetive_data = negetive_tile_data - positive_repeat_data
#     transfrom_negetive_data = transfrom_negetive_data[negetive_index[0]]
#     transform_negetive_label = np.zeros(transfrom_negetive_data.shape[0]).reshape(-1, 1)

#     return transfrom_positive_data, transform_positive_label, transfrom_negetive_data, transform_negetive_label


# def handleData_extend_mirror(positive_repeat_data, negetive_tile_data):
#     all_generate_num = positive_repeat_data.shape[0]
#     transfrom_positive_data = np.hstack( (positive_repeat_data, negetive_tile_data) )
#     transform_positive_label = np.ones(all_generate_num).reshape(-1, 1)

#     transfrom_negetive_data = np.hstack( (negetive_tile_data, positive_repeat_data) )
#     transform_negetive_label = np.zeros(all_generate_num).reshape(-1, 1)

#     return transfrom_positive_data, transform_positive_label, transfrom_negetive_data, transform_negetive_label


# def handleData_extend_not_mirror(positive_repeat_data, negetive_tile_data):
#     # 生成 label 数据，保证同一个组合不会既有正样本，又有负样本
#     all_generate_num = positive_repeat_data.shape[0]
#     init_transformed_label = np.random.randint(low=0,high=2,size=all_generate_num).reshape(-1, 1)
#     positive_index = np.where(init_transformed_label == 1)
#     negetive_index = np.where(init_transformed_label == 0)

#     transform_pos_pre_data = positive_repeat_data[positive_index[0]]
#     transform_pos_pos_data = negetive_tile_data[positive_index[0]]
#     transform_positive_label = np.ones(transform_pos_pre_data.shape[0]).reshape(-1, 1)

#     transform_neg_pre_data = negetive_tile_data[negetive_index[0]]
#     transform_neg_pos_data = positive_repeat_data[negetive_index[0]]
#     transform_negetive_label = np.zeros(transform_neg_pre_data.shape[0]).reshape(-1, 1)

#     transformed_pre_data = np.vstack( (transform_pos_pre_data, transform_neg_pre_data) )
#     transformed_pos_data = np.vstack( (transform_pos_pos_data, transform_neg_pos_data) )
#     transformed_label = np.vstack( (transform_positive_label, transform_negetive_label) )

#     seed_num = int(time.time())
#     np.random.seed(seed_num)
#     np.random.shuffle(transformed_pre_data)
#     np.random.seed(seed_num)
#     np.random.shuffle(transformed_pos_data)
#     np.random.seed(seed_num)
#     np.random.shuffle(transformed_label)
#     return transformed_pre_data, transformed_pos_data, transformed_label




# def transform_data_to_train_form(train_data_pos, train_label_neg):    
#     positive_data = train_data_pos
#     negative_data = train_label_neg
    
#     # 生成非镜像模式数据
#     length_pos = positive_data.shape[0]
#     length_neg = negative_data.shape[0]
#     all_generate_num = length_pos * length_neg

#     # repeat 每一个都连续重复
#     positive_repeat_data = np.repeat(positive_data, length_neg, axis=0)
#     # tile 整体重复
#     negetive_tile_data = np.tile(negative_data, (length_pos, 1))
    
    
#     transformed_pre_data, transformed_pos_data, transformed_label = handleData_extend_not_mirror(positive_repeat_data, negetive_tile_data)


#     return transformed_pre_data, transformed_pos_data, transformed_label


def loadTrainData(file_name):
    file_data = np.loadtxt(file_name, delimiter=',')
    label = file_data[:,-1]
    data = np.delete(file_data, -1, axis=1)
    data = data.astype(np.float64)
    label = label.reshape(-1, 1)
    label = label.astype(np.int)
    return data, label

def get_train_info(trian_method):
    train_info_list = train_method.split('_')
    model_type, sample_method, early_stop_type = train_info_list
    
    if early_stop_type == 'True':
        early_stop_type = True
        num_epochs = 5000
    else:
        num_epochs = int(early_stop_type)
        early_stop_type = False

    return model_type, sample_method, early_stop_type, num_epochs

def get_data_stat(data_numpy):
    means = np.mean(data_numpy)
    median = np.median(data_numpy)





def set_para():
    global dataset_name
    global dataset_index
    global record_index
    global device_id
    global train_method

    argv = sys.argv[1:]
    for each in argv:
        para = each.split('=')
        if para[0] == 'dataset_name':
            dataset_name = para[1]
        if para[0] == 'dataset_index':
            dataset_index = para[1]
        if para[0] == 'record_index':
            record_index = para[1]
        if para[0] == 'device_id':
            device_id = para[1]
        if para[0] == 'train_method':
            train_method = para[1]




# -------------------------------------parameters----------------------------------------
dataset_name = 'abalone19'
dataset_index = '1'
record_index = '1'
device_id = '1'
method_name = 'MLP_normal'
train_method = 'MLP_minus_notMirror_early'
num_epochs = 5000
batch_size = 50
alpha = 0.1
early_stop_type = False
# ----------------------------------set parameters---------------------------------------
set_para()
train_file_name = './test_{0}/standlization_data/{0}_std_train_{1}.csv'.format(dataset_name, dataset_index)
model_name = './test_{0}/model_{1}/record_{2}/{1}_{3}'.format(dataset_name, train_method, record_index, dataset_index)


os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)


# ----------------------------------start processing-------------------------------------
print(train_file_name)
print(model_name)
print('----------------------\n\n\n')



start = time.process_time()


base_train_data, base_train_label = loadTrainData(train_file_name)

# model_type, early_stop_type, alpha = get_train_info(train_method)
model_type, num_epochs, alpha = train_method.split('_')
num_epochs = int(num_epochs)
alpha = float(alpha)



if early_stop_type:
    # 做train 和 validation 的划分
    valid_data, valid_label, train_data, train_label = generate_valid_data(base_train_data, base_train_label)
else:
    train_data = base_train_data
    train_label = base_train_label
    valid_data = base_train_data
    valid_label = base_train_label

positive_data, negative_data = divide_data(train_data, train_label)

valid_positive_index = np.where(valid_label == 1)
valid_negative_index = np.where(valid_label == 0)

# 为了防止每次都分一次正负样例，把区分放在外边
# valid_pos_data, valid_neg_data = divide_data(valid_data, valid_label)
# transformed_valid_pre_data, transformed_valid_pos_data, transformed_valid_label = transform_data_to_train_form(valid_pos_data, valid_neg_data)
input_dim = valid_data.shape[1]


patience = 20	
# 当验证集损失在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
early_stopping = EarlyStopping(patience, verbose=True)

class Classification(nn.Module):
    def __init__(self, input_dim):
        super(Classification, self).__init__()
        self.hidden_1 = nn.Linear(input_dim, 2*input_dim)
        self.relu = nn.ReLU()
        self.output = nn.Linear(2*input_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.hidden_1(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x




net = Classification(input_dim)

init.normal_(net.hidden_1.weight, mean=0, std=0.01)
init.normal_(net.output.weight, mean=0, std=0.01)
init.constant_(net.hidden_1.bias, val=0)
init.constant_(net.output.bias, val=0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net.to(device)

loss_fn_1 = nn.BCELoss()
loss_fn_2 = nn.BCELoss()
loss_fn = nn.BCELoss()

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# input_valid_pre_data = torch.Tensor(torch.from_numpy(transformed_valid_pre_data).float())
# input_valid_pos_data = torch.Tensor(torch.from_numpy(transformed_valid_pos_data).float())
# input_valid_label = torch.Tensor(torch.from_numpy(transformed_valid_label).float())
# input_valid_pre_data = input_valid_pre_data.to(device)
# input_valid_pos_data = input_valid_pos_data.to(device)
# input_valid_label = input_valid_label.to(device)



input_valid_data = torch.Tensor(torch.from_numpy(valid_data).float())
input_valid_label = torch.Tensor(torch.from_numpy(valid_label).float())
input_valid_data = input_valid_data.to(device)
input_valid_label_gpu = input_valid_label.to(device)


for epoch in range(num_epochs):
    batch_data, batch_label = generate_normal_batch_data(train_data, train_label, batch_size)

    # input_data = torch.Tensor(torch.from_numpy(batch_data).float())
    # input_label = torch.Tensor(torch.from_numpy(batch_label).float())

    # input_data = input_data.to(device)
    # input_label = input_label.to(device)
    
    # predict = net(input_data)

    # predict = net.relu(predict - alpha)   


    batch_positive_data, batch_negative_data = divide_data(batch_data, batch_label)
    batch_positive_label = np.ones((batch_positive_data.shape[0], 1))
    batch_negative_label = np.zeros((batch_negative_data.shape[0], 1))

    # print(train_x.shape)
    input_pos_data = torch.Tensor(torch.from_numpy(batch_positive_data).float())
    input_neg_data = torch.Tensor(torch.from_numpy(batch_negative_data).float())
    input_pos_label = torch.Tensor(torch.from_numpy(batch_positive_label).float())
    input_neg_label = torch.Tensor(torch.from_numpy(batch_negative_label).float())

    input_pos_data = input_pos_data.to(device)
    input_neg_data = input_neg_data.to(device)
    input_pos_label = input_pos_label.to(device)
    input_neg_label = input_neg_label.to(device)
    
    pos_predict = net(input_pos_data)
    neg_predict = net(input_neg_data)
    neg_predict = net.relu(neg_predict-alpha)

    loss_pos = loss_fn_1(pos_predict, input_pos_label)
    loss_neg = loss_fn_2(neg_predict, input_neg_label)
    loss = loss_pos + loss_neg


    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_loss = loss.item()


    if epoch % 500 == 0:
        valid_output = net(input_valid_data)
        result =  torch.ge(valid_output, 0.5) 
        result = result.cpu()
        #计算准确率
        train_acc = accuracy_score(input_valid_label, result)

        #计算精确率
        pre = skmet.precision_score(y_true=input_valid_label, y_pred=result)

        #计算召回率
        rec = skmet.recall_score(y_true=input_valid_label, y_pred=result)
        f1 = skmet.f1_score(y_true=input_valid_label, y_pred=result)
        auc = skmet.roc_auc_score(y_true=input_valid_label, y_score=result)
        print('epoch {:.0f}, loss {:.4f}, train acc {:.2f}%, f1 {:.4f}, precision {:.4f}, recall {:.4f}, auc {:.4f}'.format(epoch+1, train_loss, train_acc*100, f1, pre, rec, auc) )
        

        
    
    if early_stop_type:
        valid_output = net(input_valid_data)
        valid_loss = torch.sum(loss_fn_1(valid_output, input_valid_label_gpu))
        early_stopping(valid_loss, net)
        # 若满足 early stopping 要求
        if early_stopping.early_stop:
            result =  torch.ge(valid_output, 0.5) 
            result = result.cpu()
            #计算准确率
            train_acc = accuracy_score(input_valid_label, result)

            #计算精确率
            pre = skmet.precision_score(y_true=input_valid_label, y_pred=result)

            #计算召回率
            rec = skmet.recall_score(y_true=input_valid_label, y_pred=result)
            f1 = skmet.f1_score(y_true=input_valid_label, y_pred=result)
            auc = skmet.roc_auc_score(y_true=input_valid_label, y_score=result)
            print('Early stopping epoch {:.0f}, loss {:.4f}, train acc {:.2f}%, f1 {:.4f}, precision {:.4f}, recall {:.4f}, auc {:.4f}\n\n\n'.format(epoch+1, train_loss, train_acc*100, f1, pre, rec, auc) )
            
            # 结束模型训练
            break
torch.save(net, model_name)


finish = time.process_time()
running_time = finish-start
print('running_time is {0}'.format(running_time))


