import glob
import re
import sys
import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import csv

def getNum(name):
    firstChar = name[-5]
    secondChar = name[-6]
    num = firstChar
    if secondChar.isdigit():
        num = secondChar + num
    return int(num)


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
    aauc = {}
    af = {}
    ap = {}
    ar = {}
    aea = {}
    aa = {}
    aer = {}
    return aauc, af, ap, ar, aea, aa, aer

def aver(l):
    return sum(l)/len(l)

def read_file(file_name):
    f = open(file_name,'r')
    for i in f:
        word,num = divide_alpha_digit(i)
        # print(word)
        # print(num)
        if word == 'the AUC is ':
            auc = num
        if word == 'the Fscore is ':
            f = num
        if word == 'the average group top precision is ':
            p = num
        if word == 'the average group recall is ':
            r = num
        if word == 'the average group top exact accuracy is ':
            ea = num
        if word == 'the average group accuracy is ':
            a = num
        if word == 'the earn rate is ':
            er = num
    return auc, f, p, r, ea, a, er


def record_diff_result(targetDic, index, result):
    if index in targetDic:
        targetDic[index].append(result)
    else:
        targetDic[index] = [result]


def calculate_average_value(target_dic):
    result_list = []
    length = len(target_dic)
    for i in range(length):
        result_list.append(target_dic[i])
    number_of_test = len(target_dic[0])
    divide_result = np.array(result_list)
    # calculate the sum result of each rows
    sum_result = np.sum(divide_result, axis=1)
    sum_result /= number_of_test
    return sum_result





def writerecord(r,l):
    for i in l:
        r.write('{0:0.4f}\t'.format(i))
    r.write('\n')

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
path = '1_year_result/'
result_dir_number = 1
record_file_name = 'result_' + str(result_number) + '_' + str(winner_number) + '.pdf'
# record_file_name = 'result_8_3.pdf'

# ----------------------------------set parameters--------------------------------------
set_para()

# ----------------------------------start processing---------------------------------

all_auc, all_fscore, all_pre, all_rec, all_exa, all_acc, all_ear = initlist()


for i in range(1,1+result_dir_number):
    goal_dir = path + 'result_' + str(result_number) + '_' + str(winner_number) + '_' + str(i) + '/*'
    # goal_dir = 'third/1_year/result_8_3_1/*'


    file_list = glob.glob(goal_dir)
    file_list.sort()
    for file_name in file_list:
        # read file get the feature and number
        auc, fscore, pre, rec, exa, acc, ear = read_file(file_name)
        # get the number of file
        num = getNum(file_name)

        # record these features into the target dictionary
        record_diff_result(all_auc, num, auc)
        record_diff_result(all_fscore, num, fscore)
        record_diff_result(all_pre, num, pre)
        record_diff_result(all_rec, num, rec)
        record_diff_result(all_exa, num, exa)
        record_diff_result(all_acc, num, acc)
        record_diff_result(all_ear, num, ear)


length = len(all_fscore)

average_fscore = calculate_average_value(all_fscore)

average_earn_rate = calculate_average_value(all_ear)

x = [i for i in range(1, length+1)]

# draw picture
# plt.figure()

# plt.scatter(x, average_fscore,s=10, c='r', marker='x')
# plt.plot(x, average_fscore, label='F_score', color='b')
# plt.plot(x, average_earn_rate, label='earn_rate', color='m')
# plt.xlabel('test week')
# plt.ylabel('Fscore')
# plt.title(record_file_name[0:-4])
# plt.legend(loc='upper left')
# plt.savefig(record_file_name)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(x, average_fscore, label='F_score', color='b')
ax2.plot(x, average_earn_rate,  label='earn_rate', color='m')
ax1.plot(np.nan, label='earn_rate', color='m')

ax1.legend(loc=0)

ax1.set_ylabel('F_score')
ax2.set_ylabel('earn_rate')
ax1.set_xlabel('test_period')

ax1.grid(linestyle='dotted')
plt.title(record_file_name[0:-4])
plt.savefig(record_file_name)
