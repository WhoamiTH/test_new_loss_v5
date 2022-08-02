# -*- coding: utf-8 -*-

# import glob
import re
import sys
import numpy as np
# import matplotlib
# matplotlib.use('agg')
# from matplotlib import pyplot as plt
import csv
import copy

class data_record_collect:
    '''
        读取结果文件夹，整理排序，生成\t分割的输出 string
    '''
    def __init__(self, dataset_path, file_name_prefix, method, data_range=5, offset=1):
        '''
            初始化，确定dataset_path等信息，以及偏移量和数据数量
        '''
        # self.auc_list = {}
        # self.f1_list = {}
        # self.precision_list = {}
        # self.recall = {}
        self.score_record = {}
        self.file_name_prefix = file_name_prefix
        self.method = method
        self.data_range = data_range
        self.offset = offset
        self.dataset_path = dataset_path
    
    def scan_file(self, file_name, data_index):
        '''
            读取文件数据，获取 [ 'Fscore', 'precision', 'recall', 'AUC' ] 指标
        '''
        # auc_str = 'the AUC is '
        # f_score_str = 'the Fscore is '
        # pre_str = 'the precision is '
        # recall_str = 'the recall is '
        try:
            # 当文件存在时
            with open(file_name,'r') as fr:
                for line in fr:
                    # 解析字符串，获取指标数据
                    line = line.replace('the ', '')
                    score_type, score_value = line.split(' is ')
                    # 获取指标字典，按照index存储
                    cur_score_record = self.score_record.get(score_type, {})
                    cur_score_record[data_index] = float(score_value)
                    self.score_record[score_type] = cur_score_record
        except:
            # 文件不存在时跳过
            pass

    def scan_all_file(self):
        '''扫描文件夹路径下所有文件'''
        for data_index in range(self.offset, self.offset+self.data_range):
            cur_file_name = self.dataset_path + self.file_name_prefix + '_{0}_pred_result.txt'.format(data_index)
            self.scan_file(cur_file_name, data_index)
    
    def get_all_valid_value(self):
        '''获取所有有效数据'''
        # score_type in [ 'Fscore', 'precision', 'recall', 'AUC' ] 指标
        if self.score_record != {}:
            for score_type in self.score_record:
                cur_score_record_dict = self.score_record[score_type]
                cur_all_result_value = []
                # 通过数据文件的范围进行检查
                for data_index in range(self.offset, self.offset+self.data_range):
                    if data_index in cur_score_record_dict:
                        cur_all_result_value.append(cur_score_record_dict[data_index])
                cur_score_record_dict['all_vaild_value_list'] = cur_all_result_value
                self.score_record[score_type] = cur_score_record_dict
        else:
            default_empty_dict = {}
            default_empty_dict['all_vaild_value_list'] = [0 for i in range(self.data_range)]
            for score_type in [ 'Fscore', 'precision', 'recall', 'AUC' ]:
                self.score_record[score_type] = default_empty_dict


    def get_all_value(self):
        '''获取所有有效数据，没有的部分补 -1'''
        # score_type in [ 'Fscore', 'precision', 'recall', 'AUC' ] 指标
        for score_type in self.score_record:
            
            cur_score_record_dict = self.score_record[score_type]
            cur_sorted_value = [-1 for i in range(self.data_range)]
            for data_index in range(self.offset, self.offset+self.data_range):
                if data_index in cur_score_record_dict:
                    # 如果存在
                    cur_sorted_value[data_index-self.offset] = cur_score_record_dict[data_index]
                    # 如果不存在， 上边已经写成 -1 了
            cur_score_record_dict['all_value_list'] = cur_sorted_value
            self.score_record[score_type] = cur_score_record_dict
        



    def get_avgerage_value(self):
        '''获取平均值，并对不存在的数据补 -1，生成排序后的list 单独存储'''
        # score_type in [ 'Fscore', 'precision', 'recall', 'AUC' ] 指标
        for score_type in self.score_record:
            cur_score_record_dict = self.score_record[score_type]
            cur_all_valie_value = cur_score_record_dict['all_vaild_value_list']
            average_value = -1
            if len(cur_all_valie_value) != 0:
                average_value = float(sum(cur_all_valie_value)) / len(cur_all_valie_value)
            cur_score_record_dict['average_value'] = average_value
            self.score_record[score_type] = cur_score_record_dict
    

    def get_max(self):
        '''获取结果最大值，并保存'''
        for score_type in self.score_record:
            cur_score_record_dict = self.score_record[score_type]
            cur_all_valie_value = cur_score_record_dict['all_vaild_value_list']
            max_value = -1
            if len(cur_all_valie_value) != 0:
                max_value = max(cur_all_valie_value)
            cur_score_record_dict['max_value'] = max_value
            self.score_record[score_type] = cur_score_record_dict
    
    def get_min(self):
        '''获取结果最小值，并保存'''
        for score_type in self.score_record:
            cur_score_record_dict = self.score_record[score_type]
            cur_all_valie_value = cur_score_record_dict['all_vaild_value_list']
            min_value = -1
            if len(cur_all_valie_value) != 0:
                min_value = min(cur_all_valie_value)
            cur_score_record_dict['min_value'] = min_value
            self.score_record[score_type] = cur_score_record_dict
    
    def get_amm_value(self):
        '''获取平均值，最大值，最小值'''
        # score_type in [ 'Fscore', 'precision', 'recall', 'AUC' ] 指标
        for score_type in self.score_record:
            cur_score_record_dict = self.score_record[score_type]
            cur_all_valie_value = cur_score_record_dict['all_vaild_value_list']
            average_value = -1
            max_value = -1
            min_value = -1
            if len(cur_all_valie_value) != 0:
                average_value = float(sum(cur_all_valie_value)) / len(cur_all_valie_value)
                max_value = max(cur_all_valie_value)
                min_value = min(cur_all_valie_value)
            cur_score_record_dict['average_value'] = average_value
            cur_score_record_dict['max_value'] = max_value
            cur_score_record_dict['min_value'] = min_value
            self.score_record[score_type] = cur_score_record_dict
    
    def get_print_str(self):
        '''按照顺序生成输出字符串，顺序是平均Fscore，最大，最小，以及全部数据'''
        self.scan_all_file() 
        self.get_all_valid_value()
        self.get_all_value()
        self.get_amm_value()

        score_type_list = [ 'Fscore', 'precision', 'recall', 'AUC' ]

        all_sorted_value = []
        for score_type in score_type_list:
            cur_score_record_dict = self.score_record[score_type]
            cur_all_value_list = cur_score_record_dict['all_value_list']
            all_sorted_value.append(cur_score_record_dict['average_value'])
            all_sorted_value.append(cur_score_record_dict['max_value'])
            all_sorted_value.append(cur_score_record_dict['min_value'])
            all_sorted_value += cur_all_value_list
        all_sorted_value = list(map(str, all_sorted_value))
        print(self.file_name_prefix)
        print(self.method)
        self.output_str = '{0}_{1}'.format(self.file_name_prefix, self.method) + '\t' + '\t'.join( all_sorted_value )
        return self.output_str



dataset_list = ['abalone19', 'ecoli1', 'glass0', 'glass5', 'pageblocks1', 'pima', 'vehicle0', 'yeast3', 'yeast5', 'yeast6']

data_range = 5
record_index = 1
model_type_list = ['LR','SVMRBF', 'SVMPOLY', 'MLP']
transform_list = ['normal']
mirror_type_list = ['normal']

# transform_list = ['concat', 'minus']
# mirror_type_list = ['Mirror', 'notMirror']
# early_stop_type_list = [ '20000', '15000', '10000', '8000', '5000', '2000']
# early stop 效果不太明显， 结果不太好
    
with open('all_dataset_normal_result.txt', 'w') as f:
    for dataset in dataset_list:
        for model_type in model_type_list:
            for transform_method in transform_list:
                for mirror_type in mirror_type_list:
                    train_method = '{0}_normal'.format(model_type)
                    test_method = 'normal'
                    cur_dataset_path = '../test_{0}/result_{1}_{2}/record_1/'.format(dataset, train_method, test_method)
                    cur_file_name_prefix = dataset
                    cur_method = '{0}_{1}'.format(train_method, test_method)
                    print(cur_dataset_path)
                    cur_obj = data_record_collect(dataset_path=cur_dataset_path, file_name_prefix=cur_file_name_prefix, method=cur_method)
                    cur_output_str = cur_obj.get_print_str()
                    f.write(cur_output_str + '\n')
                    print('end')
        f.write('\n\n\n')
























# -------------------------------------global parameters---------------------------------
# table_1 = '\\begin{table}[H]\n'
# table_1 += '\\centering\n'
# table_1 += '\\caption{the performance of different varienties of ijcai method}\n'
# table_1 += '\\label{tab:ChangingTrainData33}\n'
# table_1 += '\\begin{tabular}{|p{0.1\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|}\n'
# table_1 += '\\hline \multirow{2}{*}{Method} & \multicolumn{4}{|c|}{Original test} & \multicolumn{4}{|c|}{Border Majority} & \multicolumn{4}{|c|}{Informative Minority} & \multicolumn{4}{|c|}{Both of them}\\\\\n'
# table_1 += '\\cline{2-17} & Auc & F1 & Pre & Recall & Auc & F1 & Pre & Recall & Auc & F1 & Pre & Recall & Auc & F1 & Pre & Recall \\\\\n'
# table_1 += '\\hline IJCAI & {0:.3f} & {1:.3f} & {2:.3f}  &  {3:.3f} & {4:.3f} & {5:.3f} & {6:.3f} & {7:.3f} & {8:.3f} &  {9:.3f} & {10:.3f} &  {11:.3f}  & {12:.3f} & {13:.3f} & {14:.3f} & {15:.3f} \\\\\n'.format(ijcai_normal_normal_record.total_average_value_aauc, ijcai_normal_normal_record.total_average_value_af, ijcai_normal_normal_record.total_average_value_ap, ijcai_normal_normal_record.total_average_value_ar, ijcai_normal_bm_record.total_average_value_aauc, ijcai_normal_bm_record.total_average_value_af, ijcai_normal_bm_record.total_average_value_ap, ijcai_normal_bm_record.total_average_value_ar, ijcai_normal_im_record.total_average_value_aauc, ijcai_normal_im_record.total_average_value_af, ijcai_normal_im_record.total_average_value_ap, ijcai_normal_im_record.total_average_value_ar, ijcai_normal_both_record.total_average_value_aauc, ijcai_normal_both_record.total_average_value_af, ijcai_normal_both_record.total_average_value_ap, ijcai_normal_both_record.total_average_value_ar)
# table_1 += '\\hline Training with BM & {0:.3f} & {1:.3f} & {2:.3f}  &  {3:.3f} & {4:.3f} & {5:.3f} & {6:.3f} & {7:.3f} & {8:.3f} &  {9:.3f} & {10:.3f} &  {11:.3f}  & {12:.3f} & {13:.3f} & {14:.3f} & {15:.3f} \\\\\n'.format(ijcai_bm_normal_record.total_average_value_aauc, ijcai_bm_normal_record.total_average_value_af, ijcai_bm_normal_record.total_average_value_ap, ijcai_bm_normal_record.total_average_value_ar, ijcai_bm_bm_record.total_average_value_aauc, ijcai_bm_bm_record.total_average_value_af, ijcai_bm_bm_record.total_average_value_ap, ijcai_bm_bm_record.total_average_value_ar, ijcai_bm_im_record.total_average_value_aauc, ijcai_bm_im_record.total_average_value_af, ijcai_bm_im_record.total_average_value_ap, ijcai_bm_im_record.total_average_value_ar, ijcai_bm_both_record.total_average_value_aauc, ijcai_bm_both_record.total_average_value_af, ijcai_bm_both_record.total_average_value_ap, ijcai_bm_both_record.total_average_value_ar)
# table_1 += '\\hline Training with IM & {0:.3f} & {1:.3f} & {2:.3f}  &  {3:.3f} & {4:.3f} & {5:.3f} & {6:.3f} & {7:.3f} & {8:.3f} &  {9:.3f} & {10:.3f} &  {11:.3f}  & {12:.3f} & {13:.3f} & {14:.3f} & {15:.3f} \\\\\n'.format(ijcai_im_normal_record.total_average_value_aauc, ijcai_im_normal_record.total_average_value_af, ijcai_im_normal_record.total_average_value_ap, ijcai_im_normal_record.total_average_value_ar, ijcai_im_bm_record.total_average_value_aauc, ijcai_im_bm_record.total_average_value_af, ijcai_im_bm_record.total_average_value_ap, ijcai_im_bm_record.total_average_value_ar, ijcai_im_im_record.total_average_value_aauc, ijcai_im_im_record.total_average_value_af, ijcai_im_im_record.total_average_value_ap, ijcai_im_im_record.total_average_value_ar, ijcai_im_both_record.total_average_value_aauc, ijcai_im_both_record.total_average_value_af, ijcai_im_both_record.total_average_value_ap, ijcai_im_both_record.total_average_value_ar)
# table_1 += '\\hline Training wiht BM and IM & {0:.3f} & {1:.3f} & {2:.3f}  &  {3:.3f} & {4:.3f} & {5:.3f} & {6:.3f} & {7:.3f} & {8:.3f} &  {9:.3f} & {10:.3f} &  {11:.3f}  & {12:.3f} & {13:.3f} & {14:.3f} & {15:.3f} \\\\\n'.format(ijcai_both_normal_record.total_average_value_aauc, ijcai_both_normal_record.total_average_value_af, ijcai_both_normal_record.total_average_value_ap, ijcai_both_normal_record.total_average_value_ar, ijcai_both_bm_record.total_average_value_aauc, ijcai_both_bm_record.total_average_value_af, ijcai_both_bm_record.total_average_value_ap, ijcai_both_bm_record.total_average_value_ar, ijcai_both_im_record.total_average_value_aauc, ijcai_both_im_record.total_average_value_af, ijcai_both_im_record.total_average_value_ap, ijcai_both_im_record.total_average_value_ar, ijcai_both_both_record.total_average_value_aauc, ijcai_both_both_record.total_average_value_af, ijcai_both_both_record.total_average_value_ap, ijcai_both_both_record.total_average_value_ar)
# table_1 += '\\hline\n'

# table_1 += '\\end{tabular}\n'
# table_1 += '\\end{table}\n'




# table_2 = '\\begin{table}[H]\n'
# table_2 += '\\centering\n'
# table_2 += '\\caption{the performance of different varienties of ijcai method (AUC and F1}\n'
# table_2 += '\\label{tab:ChangingTrainData33}\n'
# table_2 += '\\begin{tabular}{|p{0.1\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|p{0.05\\textwidth}|}\n'
# table_2 += '\\hline \multirow{2}{*}{Method} & \multicolumn{2}{|c|}{Original test} & \multicolumn{2}{|c|}{Border Majority} & \multicolumn{2}{|c|}{Informative Minority} & \multicolumn{2}{|c|}{Both of them}\\\\\n'
# table_2 += '\\cline{2-9} & Auc & F1  & Auc & F1 & Auc & F1 & Auc & F1  \\\\\n'
# table_2 += '\\hline IJCAI & {0:.3f} & {1:.3f}  & {2:.3f} & {3:.3f} & {4:.3f} &  {5:.3f}  & {6:.3f} & {7:.3f} \\\\\n'.format(ijcai_normal_normal_record.total_average_value_aauc, ijcai_normal_normal_record.total_average_value_af, ijcai_normal_bm_record.total_average_value_aauc, ijcai_normal_bm_record.total_average_value_af, ijcai_normal_im_record.total_average_value_aauc, ijcai_normal_im_record.total_average_value_af, ijcai_normal_both_record
#     .total_average_value_aauc, ijcai_normal_both_record
# .total_average_value_af)
# table_2 += '\\hline Training with BM & {0:.3f} & {1:.3f}  & {2:.3f} & {3:.3f} & {4:.3f} &  {5:.3f}  & {6:.3f} & {7:.3f} \\\\\n'.format(ijcai_bm_normal_record.total_average_value_aauc, ijcai_bm_normal_record.total_average_value_af, ijcai_bm_bm_record.total_average_value_aauc, ijcai_bm_bm_record.total_average_value_af, ijcai_bm_im_record.total_average_value_aauc,ijcai_bm_im_record.total_average_value_af, ijcai_bm_both_record.total_average_value_aauc, ijcai_bm_both_record.total_average_value_af)
# table_2 += '\\hline Training with IM & {0:.3f} & {1:.3f}  & {2:.3f} & {3:.3f} & {4:.3f} &  {5:.3f}  & {6:.3f} & {7:.3f} \\\\\n'.format(ijcai_im_normal_record.total_average_value_aauc, ijcai_im_normal_record.total_average_value_af, ijcai_bm_bm_record.total_average_value_aauc, ijcai_bm_bm_record.total_average_value_af, ijcai_im_im_record.total_average_value_aauc, ijcai_im_im_record.total_average_value_af, ijcai_im_both_record.total_average_value_aauc, ijcai_im_both_record.total_average_value_af)
# table_2 += '\\hline Training wiht BM and IM & {0:.3f} & {1:.3f}  & {2:.3f} & {3:.3f} & {4:.3f} &  {5:.3f}  & {6:.3f} & {7:.3f} \\\\\n'.format(ijcai_both_normal_record.total_average_value_aauc, ijcai_both_normal_record.total_average_value_af, ijcai_both_bm_record.total_average_value_aauc, ijcai_both_bm_record.total_average_value_af, ijcai_both_im_record.total_average_value_aauc, ijcai_both_im_record.total_average_value_af, ijcai_both_both_record.total_average_value_aauc, ijcai_both_both_record.total_average_value_af)
# table_2 += '\\hline\n'

# table_2 += '\\end{tabular}\n'
# table_2 += '\\end{table}\n'

# record = open('result_table.txt','w')
# record.write(table_1)
# record.write(table_2)