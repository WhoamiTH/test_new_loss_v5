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
import tensorflow as tf
from sklearn.metrics import confusion_matrix

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# per_process_gpu_memory_fraction=0.4


# train processing


def tfrepeat(inputindex, length):
    repeatindex = tf.reshape(inputindex, [-1, 1])    # Convert to a len(yp) x 1 matrix.
    repeatindex = tf.tile(repeatindex, [1, length])  # Create multiple columns.
    repeatindex = tf.reshape(repeatindex, [-1])       # Convert back to a vector.
    return repeatindex


def tf_generate_part_data(headdataset, taildataset, headlabelset, taillabelset, headindex, tailindex):
    data_head = tf.nn.embedding_lookup(headdataset, headindex)
    data_tail = tf.nn.embedding_lookup(taildataset, tailindex)

    label_head = tf.nn.embedding_lookup(headlabelset, headindex)
    label_tail = tf.nn.embedding_lookup(taillabelset, tailindex)

    data = tf.concat([data_head, data_tail], axis=1)
    # label = tf.concat([label_head, label_tail], axis=1)
    label = label_head

    alldata = tf.concat([data, label], axis=1)
    return alldata




def set_para():
    global file_name
    global model_record_path
    global file_record_path
    global method_name
    global model_type
    global mirror_type
    global kernelpca_or_not
    global pca_or_not
    global num_of_components

    global scaler_name
    global pca_name
    global kernelpca_name
    global model_name

    argv = sys.argv[1:]
    for each in argv:
        para = each.split('=')
        if para[0] == 'file_name':
            file_name = para[1]
        if para[0] == 'model_record_path':
            model_record_path = para[1]
        if para[0] == 'file_record_path':
            file_record_path = para[1]
        if para[0] == 'method_name':
            method_name = para[1]
        if para[0] == 'model_type':
            model_type = para[1].upper()
        if para[0] == 'mirror_type':
            mirror_type = para[1]
        if para[0] == 'kernelpca':
            if para[1] == 'True':
                kernelpca_or_not = True
            else:
                kernelpca_or_not = False
        if para[0] == 'pca':
            if para[1] == 'True':
                pca_or_not = True
            else:
                pca_or_not = False
        if para[0] == 'num_of_components':
            num_of_components = int(para[1])

        if para[0] == 'scaler_name':
            scaler_name = para[1]
        if para[0] == 'pca_name':
            pca_name = para[1]
        if para[0] == 'kernelpca_name':
            kernelpca_name = para[1]
        if para[0] == 'model_name':
            model_name = para[1]

    if kernelpca_or_not and pca_or_not:
        pca_or_not = True
        kernelpca_or_not = False






# -------------------------------------parameters----------------------------------------
file_name = 'GData_train.csv'
model_record_path = '../1_year_result/model/'
file_record_path = '../1_year_result/record/'
method_name = "smote"
# model_type = 'LR'
model_type = 'SVC'
# model_type = 'DT'
# mirror_type = "mirror"
mirror_type = "not_mirror"
kernelpca_or_not = False
pca_or_not = False
num_of_components = 19

scaler_name = 'scaler.m'
pca_name = 'pca.m'
kernelpca_name = ''
model_name = 'model.m'
positive_value = 1
negative_value = -1
threshold_value = 0
winner_number = 3

train_time = 20000

# ----------------------------------set parameters---------------------------------------
set_para()

# ----------------------------------start processing-------------------------------------
print('train normal')

print(file_name)

# file_number = re.findall(r"\d+", file_name)[-1]
scaler_name = model_record_path + method_name + '_' + scaler_name
if pca_or_not:
    pca_name = model_record_path + method_name + '_' + pca_name
if kernelpca_or_not:
    kernelpca_name = model_record_path  + method_name + '_' + kernelpca_name
model_name = model_record_path + method_name + '_' + model_name

# data input
data, label = handle_data.loadTrainData(file_name)

# group_index_list = handle_data.group(data)

data = data.values
data = data.astype(np.float64)

label = label.astype(np.int)
# as there are some errors in the labels, for example, only 2 lables are left, one is 2 another one is 10
# the labels have to be transformed into a safe mode, for instance, for the case informed above, 2 will be transformed to 1, and 10 will be transformed to 2
# label = predict_test.transform_labels(label, group_index_list, winner_number)

start = clock()
new_data = handle_data.standarize_PCA_data(data, pca_or_not, kernelpca_or_not, num_of_components, scaler_name, pca_name, kernelpca_name)

train_data = new_data
batch_size = 100


positive_data, negative_data = handle_data.divide_data(train_data, label)

positive_length = positive_data.shape[0]
negative_length = negative_data.shape[0]

positive_label = handle_data.generate_transformed_label(positive_length, 1)
negative_label = handle_data.generate_transformed_label(negative_length, 0)

positive_index_list = [i for i in range(positive_length)]
negative_index_list = [i for i in range(negative_length)]


# create LogisticRegression model


single_input_size = train_data.shape[1]
transformed_input_size = 2 * single_input_size


num_class = 1

# x = tf.placeholder(tf.float32, [None, single_input_size])
x = tf.placeholder(tf.float32, [None, transformed_input_size])
y_true = tf.placeholder(tf.float32, [None, num_class])

positive_dataset = tf.Variable(positive_data, trainable=False, dtype=tf.float32)
negative_dataset = tf.Variable(negative_data, trainable=False, dtype=tf.float32)

positive_labelset = tf.Variable(positive_label, trainable=False, dtype=tf.float32)
negative_labelset = tf.Variable(negative_label, trainable=False, dtype=tf.float32)

positive_index = tf.placeholder(shape=(None), dtype=tf.int32)
negative_index = tf.placeholder(shape=(None), dtype=tf.int32)
select_negative_index = tf.placeholder(shape=(None), dtype=tf.int32)

# positive_index          = tf.reshape(positive_index, shape=(1, -1))
# negative_index          = tf.reshape(negative_index, shape=(1, -1))
# select_negative_index   = tf.reshape(select_negative_index, shape=(1, -1))


# positive positive
# data_headindex1 = tf.placeholder(shape=(None), dtype=tf.int32)
# data_tailindex1 = tf.placeholder(shape=(None), dtype=tf.int32)

data_headindex1 = tf.tile(positive_index, [tf.shape(positive_index)[0]])
data_tailindex1 = tfrepeat(positive_index, tf.shape(positive_index)[0])


# data_head1 = tf.nn.embedding_lookup(positive_dataset, data_headindex1)
# data_tail1 = tf.nn.embedding_lookup(positive_dataset, data_tailindex1)

# label_head1 = tf.nn.embedding_lookup(positive_labelset, data_headindex1)
# label_tail1 = tf.nn.embedding_lookup(positive_labelset, data_tailindex1)

# data1 = tf.concat([data_head1, data_tail1], axis=1)
# label1 = tf.concat([label_head1, label_tail1], axis=1)

# alldata1 = tf.concat([data1, label1], axis=1)

alldata1 = tf_generate_part_data(positive_dataset, positive_dataset, positive_labelset, positive_labelset, data_headindex1, data_tailindex1)


# positive negative

data_headindex2 = tf.tile(positive_index, [tf.shape(negative_index)[0]])
data_tailindex2 = tfrepeat(negative_index, tf.shape(positive_index)[0])
alldata2 = tf_generate_part_data(positive_dataset, negative_dataset, positive_labelset, negative_labelset, data_headindex2, data_tailindex2)
# data_headindex2 = tf.placeholder(shape=(None), dtype=tf.int32)
# data_tailindex2 = tf.placeholder(shape=(None), dtype=tf.int32)

# data_head2 = tf.nn.embedding_lookup(positive_dataset, data_headindex2)
# data_tail2 = tf.nn.embedding_lookup(negative_dataset, data_tailindex2)

# label_head2 = tf.nn.embedding_lookup(positive_labelset, data_headindex2)
# label_tail2 = tf.nn.embedding_lookup(negative_labelset, data_tailindex2)

# data2 = tf.concat([data_head2, data_tail2], axis=1)
# label2 = tf.concat([label_head2, label_tail2], axis=1)

# alldata2 = tf.concat([data2, label2], axis=1)

# negative positive
data_headindex3 = tf.tile(negative_index, [tf.shape(positive_index)[0]])
data_tailindex3 = tfrepeat(positive_index, tf.shape(negative_index)[0])
alldata3 = tf_generate_part_data(negative_dataset, positive_dataset, negative_labelset, positive_labelset, data_headindex3, data_tailindex3)
# data_headindex3 = tf.placeholder(shape=(None), dtype=tf.int32)
# data_tailindex3 = tf.placeholder(shape=(None), dtype=tf.int32)

# data_head3 = tf.nn.embedding_lookup(negative_dataset, data_headindex3)
# data_tail3 = tf.nn.embedding_lookup(positive_dataset, data_tailindex3)

# label_head3 = tf.nn.embedding_lookup(negative_labelset, data_headindex3)
# label_tail3 = tf.nn.embedding_lookup(positive_labelset, data_tailindex3)

# data3 = tf.concat([data_head3, data_tail3], axis=1)
# label3 = tf.concat([label_head3, label_tail3], axis=1)

# alldata3 = tf.concat([data3, label3], axis=1)

# negative negative
data_headindex4 = tf.tile(negative_index, [tf.shape(select_negative_index)[0]])
data_tailindex4 = tfrepeat(select_negative_index, tf.shape(negative_index)[0])
alldata4 = tf_generate_part_data(negative_dataset, negative_dataset, negative_labelset, negative_labelset, data_headindex4, data_tailindex4)

# data_headindex4 = tf.placeholder(shape=(None), dtype=tf.int32)
# data_tailindex4 = tf.placeholder(shape=(None), dtype=tf.int32)

# data_head4 = tf.nn.embedding_lookup(positive_dataset, data_headindex4)
# data_tail4 = tf.nn.embedding_lookup(negative_dataset, data_tailindex4)

# label_head4 = tf.nn.embedding_lookup(positive_labelset, data_headindex4)
# label_tail4 = tf.nn.embedding_lookup(negative_labelset, data_tailindex4)

# data4 = tf.concat([data_head4, data_tail4], axis=1)
# label4 = tf.concat([label_head4, label_tail4], axis=1)

# alldata4 = tf.concat([data4, label4], axis=1)


alldatafirst = tf.concat([alldata1, alldata2], axis=0)
alldatasecond = tf.concat([alldata3, alldata4], axis=0)
alldata = tf.concat([alldatafirst, alldatasecond], axis=0)

shuffled_alldata = tf.random.shuffle(alldata)


x = shuffled_alldata[:, :-1]
y_true = shuffled_alldata[:, -1]

x = tf.reshape(x, shape=(-1, transformed_input_size))
y_true = tf.reshape(y_true, shape=(-1, num_class))

hidden1 = tf.layers.dense(inputs=x, units=2*transformed_input_size, use_bias=True, activation=tf.nn.relu)


# y_pred = tf.layers.dense(inputs=hidden1, units=4, activation=tf.nn.sigmoid)
y_pred = tf.layers.dense(inputs=hidden1, units=num_class, activation=None)
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)

# print(loss)

cost = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(cost)


tf.add_to_collection('x', x)
tf.add_to_collection('y_true', y_true)
tf.add_to_collection('y_pred', y_pred)
tf.add_to_collection('cost', cost)
tf.add_to_collection('optimizer', optimizer)
saver = tf.train.Saver()

# create session and run the model

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=per_process_gpu_memory_fraction)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess = tf.Session()

sess.run(tf.global_variables_initializer())

# feed_dict_train = {
#     x         : train_data,
#     y_true    : train_label
# }
epoch_start = clock()
for i in range(train_time):
    positive_data_index, negative_data_index, select_negative_data_index = handle_data.generate_batch_data(positive_index_list, negative_index_list, positive_length, negative_length, batch_size)
    # print(train_data.shape)
    feed_dict_train = {
        positive_index : positive_data_index,
        negative_index : negative_data_index,
        select_negative_index : select_negative_data_index
    }
    # feed_dict_train = {
    #     x       : train_data,
    #     y_true  : train_label
    # }
    cost_val, true_label, pred_label, opt_obj = sess.run( [cost, y_true, y_pred, optimizer], feed_dict=feed_dict_train )
    if (i % 1000) == 0 :
        epoch_finish = clock()
        epoch_running_time = epoch_finish - epoch_start
        print('epoch: {0} cost = {1} running_time = {2}'.format(i,cost_val, epoch_running_time))
        epoch_start = clock()
#             print(pred_label)
#             print(true_label)
finish = clock()
saver.save(sess, model_name)

running_time = finish-start
print('running time is {0}'.format(running_time))