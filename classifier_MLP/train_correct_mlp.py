# train processing head
import sys
import sklearn.svm as sksvm
import sklearn.linear_model as sklin
import sklearn.tree as sktree
from sklearn.externals import joblib
from time import clock
import handle_data
import predict_test
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, f1_score
# import tensorflow as tf

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()


import torch
from torch import nn
from torch.nn import init


from sklearn.metrics import confusion_matrix

import os


def divide_data(Data, Label):
    positive_index = np.where(Label == 1)
    negative_index = np.where(Label == 0)

    positive = Data[positive_index[0]]
    negative = Data[negative_index[0]]
    return positive, negative

def generate_valid_data(data, label, size=0.05):
    data_length = data.shape[0]
    valid_length = int( data_length * size )

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
    if len(train_info_list) == 2:
        model_type, early_stop_type = train_info_list
        mirror_type = 'normal'
    
    if len(train_info_list) == 3:
        model_type, mirror_type, early_stop_type = train_info_list
    
    if early_stop_type == 'True':
        early_stop_type = True
        num_epochs = 5000
    else:
        num_epochs = int(early_stop_type)
        early_stop_type = False


    return model_type, mirror_type, early_stop_type, num_epochs



# train processing

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

train_data, train_label = loadTrainData(train_file_name)

model_type, mirror_type, early_stop_type, num_epochs = get_train_info(train_method)



if early_stop_type:
    valid_data, valid_label, train_data, train_label = generate_valid_data(train_data, train_label)
else:
    valid_data = train_data
    valid_label = train_label
positive_data, negative_data = divide_data(train_data, train_label)

if transform_method == 'normal':
    transformed_valid_data, transformed_valid_label = transform_data_to_train_form(transform_method, mirror_type, valid_data, valid_label)
else:    
    valid_pos_data, valid_neg_data = divide_data(valid_data, valid_label)
    transformed_valid_data, transformed_valid_label = transform_data_to_train_form(transform_method, mirror_type, valid_pos_data, valid_neg_data)
input_dim = transformed_valid_data.shape[1]


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
        x = self.sigmoid(x)
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

loss = nn.BCELoss()  

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)


input_valid_data = torch.Tensor(torch.from_numpy(transformed_valid_data).float())
input_valid_label = torch.Tensor(torch.from_numpy(transformed_valid_label).float())
input_valid_data = input_valid_data.to(device)





train_data = train_data.astype(np.float64)
train_label = train_label.astype(np.int)

start = clock()
train_data = handle_data.standarize_PCA_data(train_data, pca_or_not, kernelpca_or_not, num_of_components, scaler_name, pca_name, kernelpca_name)

# train_data = new_data
batch_size = 50


positive_data, negative_data = handle_data.divide_data(train_data, train_label)
input_dim = train_data.shape[1]

# create LogisticRegression model




# def handleData_extend_not_mirror(positive_data, negative_data, positive_value=1, negative_value=0):
#     # 生成非镜像模式数据
#     tem_pre = []
#     tem_pos = []
#     teml = []
#     length_pos = len(positive_data)
#     length_neg = len(negative_data)
#     all_generate_num = length_pos * length_neg
#     random_label_array = np.random.randint(low=0,high=2,size=all_generate_num)
#     random_label_list = random_label_array.tolist()
#     label_index = 0
#     for pos_index in range(length_pos):
#         for neg_index in range(length_neg):
#             cur_label = random_label_list[label_index]
#             label_index += 1
#             if int(cur_label) == 1:
#                 tem_pre.append(positive_data[pos_index])
#                 tem_pos.append(negative_data[neg_index])
#                 teml.append( [positive_value] )
#             else:
#                 tem_pos.append(positive_data[pos_index])
#                 tem_pre.append(negative_data[neg_index])
#                 teml.append( [negative_value] )
#     return tem_pre, tem_pos, teml


def handleData_extend_not_mirror(positive_data, negative_data, positive_value=1, negative_value=0):
    # 生成非镜像模式数据，在这个场景中，如果使用镜像数据，相同的样本会被主动循环多次，而不是batch随机获取，容易造成overfitting
    # 所以 只使用非镜像数据
    tem_pre = []
    tem_pos = []
    teml = []
    length_pos = len(positive_data)
    length_neg = len(negative_data)
    # all_generate_num = length_pos * length_neg
    all_generate_num = length_pos
    random_label_array = np.random.randint(low=0,high=2,size=all_generate_num)
    random_label_list = random_label_array.tolist()
    label_index = 0
    for pos_index in range(length_pos):
        cur_label = random_label_list[label_index]
        label_index += 1
        if int(cur_label) == 1:
            tem_pre.append(positive_data[pos_index])
            tem_pos.append(negative_data[pos_index])
            teml.append( [positive_value] )
        else:
            tem_pre.append(negative_data[pos_index])
            tem_pos.append(positive_data[pos_index])
            teml.append( [negative_value] )
    return tem_pre, tem_pos, teml


def generate_batch_data(positive_data, negative_data, batch_size):
    positive_length = positive_data.shape[0]
    negative_length = negative_data.shape[0]

    if batch_size > positive_length:
        batch_size = positive_length

    positive_data_index = np.random.choice(positive_length, batch_size, replace=False)
    negative_data_index = np.random.choice(negative_length, batch_size, replace=False)

    current_positive_data = positive_data[positive_data_index]
    current_negative_data = negative_data[negative_data_index]

    train_data_pre, train_data_pos, train_label = handleData_extend_not_mirror(current_positive_data, current_negative_data)
    train_data_pre = np.array(train_data_pre)
    train_data_pos = np.array(train_data_pos)
    train_label = np.array(train_label).reshape(-1, 1)
    return train_data_pre, train_data_pos, train_label

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


net = Classification(input_dim)

init.normal_(net.hidden_1.weight, mean=0, std=0.01)
init.normal_(net.output.weight, mean=0, std=0.01)
init.constant_(net.hidden_1.bias, val=0)
init.constant_(net.output.bias, val=0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# if torch.cuda.device_count() > 1:
#     net = nn.DataParallel(net) # 包装为并行风格模型
net.to(device)

loss = nn.BCELoss()  

optimizer = torch.optim.SGD(net.parameters(), lr=1.2, momentum=0.9)

def evaluate_accuracy(x, y, net):
    out = net(x)
    # result = out.numpy()
    # result[result<0.5] = 0
    # result[result>=0.5] = 1
    result =  torch.ge(out, 0.5) 
    #计算准确率
    Accuracy=accuracy_score(y, result)

    #计算精确率
    Precision=precision_score(y, result, average='macro')

    #计算召回率
    Recall=recall_score(y, result, average='macro')
    F1 = f1_score(y_true=y, y_pred=result)

    # print('accuracy is {0}'.format(Accuracy))
    # print('Precision is {0}'.format(Precision))
    # print('Recall is {0}'.format(Recall))
    # print('F1 is {0}'.format(F1))
    # correct = result.eq(y)
    # correct = correct.sum().item()
    # n = y.shape[0]
    # return correct/n
    return Accuracy, Precision, Recall, F1

for epoch in range(num_epochs):
    train_pre, train_pos, train_y = generate_batch_data(positive_data, negative_data, batch_size)
    input_data_pre = torch.Tensor(torch.from_numpy(train_pre).float())
    input_data_pos = torch.Tensor(torch.from_numpy(train_pos).float())

    train_label = torch.Tensor(torch.from_numpy(train_y).float())

    input_data_pre = input_data_pre.to(device)
    input_data_pos = input_data_pos.to(device)
    train_label = train_label.to(device)

    out_pre = net(input_data_pre)
    # out_pos = net(input_data_pos)

    if epoch == num_epochs-1:
        for i in range(len(out_pre)):
            print(train_label[i].item(), out_pre[i].item())


    # transformed_pre = net.relu(out_pre-out_pos)    
    
    l = loss(out_pre, train_label)
    optimizer.zero_grad()
    l.backward()
    optimizer.step()
    train_loss = l.item()

    if epoch % 100 == 0:
        train_acc, pre, rec, f1 = evaluate_accuracy(input_data_pre, train_label, net)
        print('epoch {:.0f}, loss {:.4f}, train acc {:.2f}%, f1 {:.4f}, precision {:.4f}, recall {:.4f}'.format(epoch+1, train_loss, train_acc*100, f1, pre, rec) )


torch.save(net, model_name)
