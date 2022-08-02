import glob
import re
import sys
import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import csv
import copy
import random

def plotimage(self):
    # draw picture
    length = self.af.shape[0]
    x = [i for i in range(1, length+1)]
    # print(self.average_af)
    # print(self.af)
    plt.figure()

    # plt.scatter(x, self.af, s=10, c='r', marker='x')
    plt.plot(x, self.af[:,0], label='first', color='b')
    plt.plot(x, self.af[:,1], label='second', color='cyan')
    plt.plot(x, self.af[:,2], label='third', color='yellow')
    plt.plot(x, self.af[:,3], label='fourth', color='g')

    # self.af = self.af.reshape(-1,1)
    # number_of_result = self.af.shape[0]
    # sum_result = np.sum(self.af, axis=0)
    # sum_result /= number_of_result
    # # print(sum_result)
    # average_af = [sum_result for i in range(1, length+1)]

    plt.plot(x, self.average_af, label='average_F', c='r')

    plt.xlabel('test week')
    plt.ylabel('Fscore')
    plt.legend(loc='upper left')
    plt.savefig(self.name + '.pdf')

def plotimage(self):
    # draw picture
    length = self.af.shape[0]
    num_of_test = self.af.shape[1]
    x = [i for i in range(1, length+1)]
    average_line = [self.total_average_value_af for i in range(1,length+1)]

    fig, (ax1, ax2) = plt.subplots(1,2)
    if num_of_test > 0:
        ax1.plot(x, self.af[:,0], label='first', color='b')
    if num_of_test > 1:
        ax1.plot(x, self.af[:,1], label='second', color='cyan')
    if num_of_test > 2:
        ax1.plot(x, self.af[:,2], label='third', color='yellow')
    if num_of_test > 3:
        ax1.plot(x, self.af[:,3], label='fourth', color='g')
    ax1.set_title('divided result')

    ax2.plot(x, self.row_average_value_af, label='average_F', c='r')
    ax2.plot(x, average_line, label='average_line', c='purple')
    ax2.set_title('average result')

    ax1.set_xlabel('test fold')
    ax1.set_ylabel('Fscore')

    ax2.set_xlabel('test fold')
    ax2.set_ylabel('Fscore')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')
    plt.savefig(self.name + '.pdf')



# def set_para():
#     global path
#     global result_number
#     global winner_number
#     global record_file_name
#     global result_dir_number

#     argv = sys.argv[1:]
#     for each in argv:
#         para = each.split('=')
#         if para[0] == 'path':
#             path = para[1]
#         if para[0] == 'result_number':
#             result_number = int(para[1])
#         if para[0] == 'winner_number':
#             winner_number = int(para[1])
#         if para[0] == 'record_file_name':
#             record_file_name = para[1]
#         if para[0] == 'result_dir_number':
#             result_dir_number = int(para[1])

def get_estimate_result(x, y):
    t = 2*x - 3
    e_t = random.uniform(-0.3, 0.3)
    t += t
    if y >= t:
        return 0
    else:
        return 1

pos_instance_list = []
neg_instance_list = []
label_list = []
for x in range(1000):
    x = random.uniform(-10, 10)
    y = random.uniform(-10, 10)
    label = get_estimate_result(x, y)
    if label == 1:
        pos_instance_list.append([x, y])
    else:
        neg_instance_list.append([x, y])










# # -------------------------------------global parameters---------------------------------
# # result_number = 3
# # winner_number = 3
# # path = './1_year_result/'
# # result_dir_number = 1
# record_file_name = 'result_' + str(result_number) + '_' + str(winner_number) + '.pdf'
# # record_file_name = 'result_8_3.pdf'

# # ----------------------------------set parameters--------------------------------------
# # set_para()

# # ----------------------------------start processing---------------------------------
# # path = '../1_year_result/record_1/'
# # total_record_number = 4




# ijcai_normal_normal_name  = 'ijcai_normal_normal'




# ijcai_normal_normal_record  = different_percent_datarecord(ijcai_normal_normal_name, path, total_record_number)


# ijcai_normal_normal_record.read_file_list()


# ijcai_normal_normal_record.calculate_average()



# ijcai_normal_normal_record.plotimage()



# print('Fscore {0}'.format(ijcai_normal_normal_record.total_average_value_af))
# print('AUC {0}'.format(ijcai_normal_normal_record.total_average_value_aauc))
