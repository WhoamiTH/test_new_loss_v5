

import sys
import os

dataset_list = ['abalone19', 'ecoli1', 'glass0', 'glass5', 'pageblocks1', 'pima', 'vehicle0', 'yeast3', 'yeast5', 'yeast6']
# dataset_list = ['ecoli1', 'glass0', 'glass5', 'pageblocks1', 'yeast3', 'yeast5', 'yeast6']


data_range = 5
record_index = 1
bash_file_name_prefix = 'train_mlp_'

early_stop_type_list = [ '30000', '25000', '20000', '15000', '10000', '8000', '5000', '2000']

alpha_list = []
# for i in range(10):
#     alpha_list.append(float(i)/100)

for i in range(0, 40, 1):
    alpha_list.append(float(i)/100)

# early stop 效果不太明显， 结果不太好

threshold_list = []

for i in range(0, 50, 1):
    threshold_list.append(float(i)/100)

# for file_index in dataset_dict:
    # dataset_list = dataset_dict[file_index]

command_list = []
for dataset in dataset_list:
    for early_stop_type in early_stop_type_list:
        for alpha in alpha_list:
            # for threshold in threshold_list:
            threshold = 0.5
            cur_command_list = []
            train_method = 'MLP_{0}_{1}'.format(early_stop_type, alpha)
            test_method = 'normal_{0}'.format(threshold)

            cur_path = './test_{0}/result_{1}_{2}/'.format(dataset, train_method, test_method)                
            if os.path.exists(cur_path):
                cur_command_list.append('rm -rf ./test_{0}/result_{1}_{2}/\n'.format(dataset, train_method, test_method))  
                    
            cur_path = './test_{0}/model_{1}/'.format(dataset, train_method)
            if os.path.exists(cur_path):
                cur_command_list.append('rm -rf ./test_{0}/model_{1}/\n'.format(dataset, train_method)) 
    
            if len(cur_command_list) != 0:
                cur_command_list.append('\n\n\n')
                command_list.append(cur_command_list)    

with open('del_file.sh', 'w') as fsh:
    fsh.write('#!/bin/bash\n')
    fsh.write('set -e\n\n\n')      

    for item_command_list in command_list:
        for line in item_command_list:
            fsh.write(line)
   