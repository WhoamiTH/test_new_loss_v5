#@title handle_data
# handle data head
import numpy as np
import random
import sklearn.preprocessing as skpre
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.externals import joblib
import pandas as pd


# handle data function begining

def loadTrainData(file_name):
    file_data = pd.read_csv(file_name, header=None)
    data = file_data.values
    label = data[:,-1]
    data = np.delete(data, 0, axis=1)
    data = np.delete(data, -1, axis=1)
    # data = np.delete(data, 0, axis=1)
    data = data.astype(np.float64)
    data = pd.DataFrame(data)
    return data, label

# def loadTrainData(file_name):
#     tem = np.loadtxt(file_name, dtype=np.float, delimiter=',')
#     label = tem[:,-1]
#     data = tem[:, 1:-1]
#     # data = data.astype(np.float)
#     # label = label.astype(np.int)
#     return data, label

def loadTestData(file_name):
    # the original data should be saved because the new result will be append to the original data to form the new data
    file_data = pd.read_csv(file_name,header=None)
    data = file_data.values
    data = data[:,1:]
    # data = np.delete(data, 0, axis=1)
    data.astype(np.float64)
    data = pd.DataFrame(data)
    return file_data, data

# def next_batch(positive_data, negative_data, batch_size=1000, pairsize=2, seq_length=1):
#     x_examples = []
#     y_examples = []
#     t_examples = []
    
#     positive_length = positive_data.shape[0]
#     negative_length = negative_data.shape[0]


#     for i in range(batch_size):
#         posi_or_nega = random.randint(0, 1)
#         if posi_or_nega > 0:
#             positive_idx = random.randint(0, positive_length - seq_length)
#             x_examples.append(positive_data[positive_idx : positive_idx + seq_length])
#             y_examples.append([1])
#             negative_idx = random.randint(0, negative_length - seq_length)
#             x_examples.append(negative_data[negative_idx : negative_idx + seq_length])
#             y_examples.append([0])
#             t_examples.append([1])
#         else:
#             negative_idx = random.randint(0, negative_length - seq_length)
#             x_examples.append(negative_data[negative_idx : negative_idx + seq_length])
#             y_examples.append([0])
#             positive_idx = random.randint(0, positive_length - seq_length)
#             x_examples.append(positive_data[positive_idx : positive_idx + seq_length])
#             y_examples.append([1])
#             t_examples.append([0])
#     return x_examples, y_examples, t_examples

def next_batch(positive_data, negative_data, pairsize=2, seq_length=1):
    x_examples = []
    y_examples = []
    t_examples = []
    
    positive_length = positive_data.shape[0]
    negative_length = negative_data.shape[0]


    # for i in range(batch_size):
    posi_or_nega = random.randint(0, 1)
    if posi_or_nega > 0:
        positive_idx = random.randint(0, positive_length - seq_length)
        x_examples.append(positive_data[positive_idx : positive_idx + seq_length])
        y_examples.append([1])
        negative_idx = random.randint(0, negative_length - seq_length)
        x_examples.append(negative_data[negative_idx : negative_idx + seq_length])
        y_examples.append([0])
        t_examples.append([1])
    else:
        negative_idx = random.randint(0, negative_length - seq_length)
        x_examples.append(negative_data[negative_idx : negative_idx + seq_length])
        y_examples.append([0])
        positive_idx = random.randint(0, positive_length - seq_length)
        x_examples.append(positive_data[positive_idx : positive_idx + seq_length])
        y_examples.append([1])
        t_examples.append([0])
    return x_examples, y_examples, t_examples


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


def transform_data_to_test_form_data(test_data, reference_data):
    temd = []
    length_test_data = len(test_data)
    reference_data_list = list(reference_data)
    reference_samples = random.sample(reference_data_list, length_test_data)
    reference_samples = np.array(reference_samples)
    transformed_test_data = np.hstack((test_data, reference_samples))
    return transformed_test_data




def handleData_extend(Data_pre, Data_pos, Label_pre, Label_pos):
    temd = []
    teml = []
    length_pre = len(Data_pre)
    length_pos = len(Data_pos)
    for j in range(length_pre):
        for t in range(length_pos):
            temd.append(data_extend(Data_pre[j], Data_pos[t]))    
    teml = np.zeros((length_pre*length_pos,4))
    if Label_pre == 1:
        teml[:,0] = 1
        teml[:,1] = 0
    else:
        teml[:,0] = 0
        teml[:,1] = 1
    if Label_pos == 1:
        teml[:,2] = 1
        teml[:,3] = 0
    else:
        teml[:,2] = 0
        teml[:,3] = 1
    return temd, teml


def transform_data_to_compare_data(Data_posi, Data_nega):
    tem_data = []
    tem_label = []
    
    # print(Data_nega.shape)
    # print(Data_posi.shape)

    length_posi = Data_posi.shape[0]
    length_nega = Data_nega.shape[0]
    tem_data_nega_list = list(Data_nega)

    data_nega = random.sample(tem_data_nega_list, length_posi)
    data_nega = np.array(data_nega)
    temd, teml = handleData_extend(Data_posi, Data_posi,1,1)
    tem_data.extend(temd)
    tem_label.extend(teml)
    del(temd)
    del(teml)
    
    
    temd, teml = handleData_extend(Data_posi, data_nega,1,0)
    tem_data.extend(temd)
    tem_label.extend(teml)
    del(temd)
    del(teml)
    
    temd, teml = handleData_extend(data_nega, Data_posi,0,1)
    tem_data.extend(temd)
    tem_label.extend(teml)
    del(temd)
    del(teml)
    
#     undersampling_nega_index = np.random.choice(length_nega, length_posi, replace=False)
#     undersampling_nega_data = Data_nega[undersampling_nega_index]
    temd, teml = handleData_extend(Data_nega, data_nega,0,0)
    tem_data.extend(temd)
    tem_label.extend(teml)
    del(temd)
    del(teml)

    data = np.array(tem_data)
    label = np.array(tem_label)
    
    rng_state = np.random.get_state()
    tem_data = np.random.shuffle(data)
    np.random.set_state(rng_state)
    tem_label = np.random.shuffle(label)



    return data, label

def generate_batch_data(positive_data, negative_data, batch_size):
    positive_length = positive_data.shape[0]
    negative_length = negative_data.shape[0]

    times = negative_length / positive_length
    times = round(times)

    if times>3 :
        times = 3

    # print(times)

    positive_data_index = np.random.choice(positive_length, batch_size, replace=False)
    negative_data_index = np.random.choice(negative_length, times*batch_size, replace=False)

    current_positive_data = positive_data[positive_data_index]
    current_negative_data = negative_data[negative_data_index]

    train_data, train_label = transform_data_to_compare_data(current_positive_data, current_negative_data)
    return train_data, train_label



# def divide_data(Data, Label):
#     length = Data.shape[0]
#     positive = []
#     negative = []
#     for i in range(length):
#         if Label[i] == 1:
#             positive.append(Data[i])
#         else:
#             negative.append(Data[i])
#     positive = np.array(positive)
#     negative = np.array(negative)
#     return positive, negative


def divide_data(Data, Label):
    # length = Data.shape[0]
    # positive = []
    # negative = []
    # for i in range(length):
    #     if Label[i] == 1:
    #         positive.append(Data[i])
    #     else:
    #         negative.append(Data[i])
    positive_index = np.where(Label == 1)
    negative_index = np.where(Label == 0)

    positive = Data[positive_index[0]]
    negative = Data[negative_index[0]]
    return positive, negative



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







# handle data functions ending
