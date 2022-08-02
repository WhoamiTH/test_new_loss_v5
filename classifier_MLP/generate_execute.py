import sys


method_name = 'change_loss'
total_record_number = 4
test_number = 5
winner_number = 3
file_name_pre = 'glass0'

# record_number = 1
i = 1
file_record_index = 1

train_record_name = 'execute_correct_'
# file = open(record_name,'w')
for record_number in range(1, test_number+1):
    with open(train_record_name+str(record_number)+'.sh','w') as train_file:
        # i = 1
        """
        train parameter list:
        file_name
        model_record_path
        file_record_path
        method_name
        scaler_name
        pca_or_not
        pca_name
        model_name
        """
        train_file.write('python train_correct.py file_name=../1_year_data/{7}_train_{1}.csv model_record_path=../1_year_result/model_{2}/ file_record_path=../1_year_result/record_{2}/ method_name={3} scaler_name=scaler_{4}.m pca=False pca_name=pca_{5}.m model_name=model_{6}/my_model.pkl\n'.format(record_number, record_number, file_record_index, method_name, record_number, record_number, record_number, file_name_pre))
        """
        test parameter list:
        file_name
        model_record_path
        file_record_path
        method_name
        scaler_name
        pca_name
        model_name
        record_name
        """
        train_file.write('python test.py train_file_name=../1_year_data/{8}_train_{1}.csv test_file_name=../1_year_data/{8}_test_origin_{1}.csv model_record_path=../1_year_result/model_{2}/ file_record_path=../1_year_result/record_{2}/ method_name={3} scaler_name=scaler_{4}.m model_name=model_{6}/my_model.pkl record_name=validation_result_{7}.csv\n'.format(record_number, record_number, file_record_index, method_name, record_number, record_number, record_number, record_number, file_name_pre))



# test_record_name = 'execute_test.sh'
# with open(test_record_name, 'w') as test_file:
#     # i = 1
#     """
#     test parameter list:
#     file_name
#     model_record_path
#     file_record_path
#     method_name
#     scaler_name
#     pca_name
#     model_name
#     record_name
#     """
#     test_file.write('python test.py train_file_name=../1_year_data/{8}_train.csv test_file_name=../1_year_data/{8}_test.csv model_record_path=../1_year_result/model_{1}/ file_record_path=../1_year_result/record_{2}/ method_name={3} scaler_name=scaler_{4}.m model_name=model_{6} record_name=test_result_{7}.csv\n'.format(i, record_number, record_number, method_name, i, i, i, i, file_name_pre))

'''

for record_number in range(1, total_record_number+1):
    record_name = 'execute_{0}.sh'.format(record_number)
    file = open(record_name,'w')
    for i in range(1, 1+test_number):
        # i = 0
        """
        train parameter list:
        file_name
        model_record_path
        file_record_path
        method_name
        scaler_name
        pca_or_not
        pca_name
        model_name
        """
        file.write('python train.py file_name=../1_year_data/{7}_train_{0}.csv model_record_path=../1_year_result/model_{1}/ file_record_path=../1_year_result/record_{2}/ method_name={3} scaler_name=scaler_{4}.m pca=False pca_name=pca_{5}.m model_name=model_{6}/my_model\n'.format(i, record_number, record_number, method_name, i, i, i,file_name_pre))
        """
        test parameter list:
        file_name
        model_record_path
        file_record_path
        method_name
        scaler_name
        pca_name
        model_name
        record_name
        """
        file.write('python test.py train_file_name=../1_year_data/{8}_train_{0}.csv test_file_name=../1_year_data/{8}_test_{0}.csv model_record_path=../1_year_result/model_{1}/ file_record_path=../1_year_result/record_{2}/ method_name={3} scaler_name=scaler_{4}.m model_name=model_{6} record_name=result_{7}.csv\n'.format(i, record_number, record_number, method_name, i, i, i, i, file_name_pre))
        """
        analyse parameter list:
        predict_label_file_name
        true_label_file_name
        model_record_path
        file_record_path
        method_name
        winner_number
        """
        file.write('python analyse_result.py predict_label_file_name={0}_result_{1}.csv true_label_file_name={8}_test_origin_{2}.csv model_record_path=../1_year_result/model_{3}/ file_record_path=../1_year_result/record_{4}/ method_name={5} winner_number={6} record_name=result_{7}.txt\n'.format(method_name, i, i, record_number, record_number, method_name, winner_number, i, file_name_pre))

    file.close()
'''