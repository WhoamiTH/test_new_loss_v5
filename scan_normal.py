import glob
import re
import sys
import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import csv
import copy

class datarecord:
    def __init__(self, name):
        self.aauc = {}
        self.af = {}
        self.ap = {}
        self.ar = {}
        self.aea = {}
        self.aa = {}
        self.aer = {}
        self.name = name
        self.current_file_number = -1

    def getNum(self, file_name):
        firstChar = file_name[-5]
        secondChar = file_name[-6]
        num = firstChar
        if secondChar.isdigit():
            num = secondChar + num
        self.current_file_number = int(num)

    def record_diff_result(self, targetDic, index, result):
        if index in targetDic:
            targetDic[index].append(result)
        else:
            targetDic[index] = [result]

    def check_category(self, current_str, target_str, target_dic):
        if (current_str.find(target_str) != -1):
            i = current_str.replace(target_str, '')
            target_number = float(i)
            self.record_diff_result(target_dic, self.current_file_number, target_number)
   
    def read_file(self, file_name):
        self.getNum(file_name)
        f = open(file_name,'r')
        auc_str = 'the AUC is '
        f_score_str = 'the Fscore is '
        pre_str = 'the average group top precision is '
        recall_str = 'the average group recall is '
        exact_accu_str = 'the average group top exact accuracy is '
        group_accu_str = 'the average group accuracy is '
        er_str = 'the earn rate is '

        for i in f:
            self.check_category(i, auc_str, self.aauc)
            self.check_category(i, f_score_str, self.af)
            self.check_category(i, pre_str, self.ap)
            self.check_category(i, recall_str, self.ar)
            self.check_category(i, exact_accu_str, self.aea)
            self.check_category(i, group_accu_str, self.aa)
            self.check_category(i, er_str, self.aer)


    def calculate_average(self):
        self.average_af = self.calculate_average_value(self.af)

    def calculate_average_value(self, target_dic):
        key_list = []
        result_list = []
        length = len(target_dic)
        # print(target_dic)
        for key in target_dic:
            key_list.append(key)
        key_list.sort()
        for key in key_list:
            result_list.append(target_dic[key])
        key = key_list[0]
        number_of_test = len(target_dic[key])
        divide_result = np.array(result_list)
        self.af = divide_result
        # calculate the sum result of each rows
        sum_result = np.sum(divide_result, axis=1)
        sum_result /= number_of_test
        return sum_result

    def get_max(self):
        print(self.name, np.where(self.average_af==self.average_af.max()), self.average_af.max())

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




class different_percent_datarecord(datarecord):
    def __init__(self, name, path, total_record_number):
        datarecord.__init__(self, name)
        self.path = path
        self.total_record_number = total_record_number

    def read_file_list(self):
        for record_number in range(1, total_record_number+1):
            record_path = path + self.name + '/record_{0}/*.txt'.format(record_number)
            file_list = glob.glob(record_path)
            file_list.sort()
            for file_name in file_list:
                if file_name.find(self.name) != -1:
                    self.read_file(file_name)

    def calculate_average(self):
        print(self.name)
        self.print_average()
        self.row_average_value_af, self.total_average_value_af, self.af = self.calculate_average_value(self.af)
        # print('average_af is {0}'.format(self.average_af))
        self.row_average_value_ap, self.total_average_value_ap, self.ap = self.calculate_average_value(self.ap)
        # print('average_ap is {0}'.format(self.average_ap))
        self.row_average_value_ar, self.total_average_value_ar, self.ar = self.calculate_average_value(self.ar)
        # print('average_ar is {0}'.format(self.average_ar))
        self.row_average_value_aauc, self.total_average_value_aauc, self.aauc = self.calculate_average_value(self.aauc)
        # print('average_aauc is {0}'.format(self.average_aauc))
        # self.af = self.calculate_average_af(self.af)

    def calculate_average_af(self, target_dic):
        key_list = []
        result_list = []
        length = len(target_dic)
        # print(target_dic)
        for key in target_dic:
            key_list.append(key)
        # print(key_list)
        key_list.sort()
        for key in key_list:
            result_list.append(target_dic[key])
        # print(key_list)
        key = key_list[0]
        number_of_test = len(target_dic[key])
        divide_result = np.array(result_list)
        return divide_result

    def calculate_average_value(self, target_dic):
        key_list = []
        result_list = []
        length = len(target_dic)
        # print(target_dic)
        for key in target_dic:
            key_list.append(key)
        # print(key_list)
        key_list.sort()
        for key in key_list:
            result_list.append(target_dic[key])
        # print(key_list)
        key = key_list[0]
        number_of_item_test = len(target_dic[key])
        total_test = number_of_item_test * length
        divide_result = np.array(result_list)
        # self.af = divide_result
        # calculate the sum result of each rows
        rowsum_result = np.sum(divide_result, axis=1)
        row_average_value = rowsum_result/number_of_item_test
        total_sum = np.sum(rowsum_result, axis=0)
        total_average_value = total_sum/total_test
        return row_average_value, total_average_value, divide_result

    def print_average(self):
        print(self.name)
        print_average_aauc = self.calculate_print_average_value(self.aauc)
        print('average_aauc is {0}'.format(print_average_aauc))
        print_average_af = self.calculate_print_average_value(self.af)
        print('average_af is {0}'.format(print_average_af))
        print_average_ap = self.calculate_print_average_value(self.ap)
        print('average_ap is {0}'.format(print_average_ap))
        print_average_ar = self.calculate_print_average_value(self.ar)
        print('average_ar is {0}'.format(print_average_ar))
        

    def calculate_print_average_value(self, target_dic):
        key_list = []
        result_list = []
        length = len(target_dic)
        # print(target_dic)
        for key in target_dic:
            key_list.append(key)
        # print(key_list)
        key_list.sort()
        for key in key_list:
            result_list.append(target_dic[key])
        # print(key_list)
        # key = key_list[0]
        number_of_test = len(key_list)
        number_of_test *= length
        divide_result = np.array(result_list)
        # calculate the sum result of each rows
        # sum_result = np.sum(divide_result, axis=1)
        # calculate the sum result of each columns
        # print(divide_result)
        sum_result = np.sum(divide_result, axis=0)
        sum_result = np.sum(sum_result, axis=0)
        # print(sum_result)
        # print(number_of_test)
        sum_result /= number_of_test
        # print(sum_result)
        return sum_result

    def calculate_average_line(self):
        number_of_test = self.average_af.shape[0]
        local_average_af = copy.deepcopy(self.average_af)
        sum_result = np.sum(self.average_af, axis=0)
        sum_result /= number_of_test
        self.average_line = sum_result

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



def set_para():
    global path
    global result_number
    global winner_number
    global record_file_name
    global result_dir_number

    argv = sys.argv[1:]
    for each in argv:
        para = each.split('=')
        if para[0] == 'path':
            path = para[1]
        if para[0] == 'result_number':
            result_number = int(para[1])
        if para[0] == 'winner_number':
            winner_number = int(para[1])
        if para[0] == 'record_file_name':
            record_file_name = para[1]
        if para[0] == 'result_dir_number':
            result_dir_number = int(para[1])







# -------------------------------------global parameters---------------------------------
result_number = 3
winner_number = 3
path = './1_year_result/'
result_dir_number = 1
record_file_name = 'result_' + str(result_number) + '_' + str(winner_number) + '.pdf'
# record_file_name = 'result_8_3.pdf'

# ----------------------------------set parameters--------------------------------------
set_para()

# ----------------------------------start processing---------------------------------
# path = '../1_year_result/record_1/'
total_record_number = 4




ijcai_normal_normal_name  = 'ijcai_normal_normal'




ijcai_normal_normal_record  = different_percent_datarecord(ijcai_normal_normal_name, path, total_record_number)


ijcai_normal_normal_record.read_file_list()


ijcai_normal_normal_record.calculate_average()



ijcai_normal_normal_record.plotimage()



print('Fscore {0}'.format(ijcai_normal_normal_record.total_average_value_af))
print('AUC {0}'.format(ijcai_normal_normal_record.total_average_value_aauc))
