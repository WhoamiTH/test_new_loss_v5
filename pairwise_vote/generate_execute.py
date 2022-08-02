import sys


method_name = 'pairwise_vote'
total_record_number = 5
test_number = 5
winner_number = 3
dataset_name = 'abalone19'
for record_number in range(1, total_record_number+1):
    record_name = 'execute_{0}.sh'.format(record_number)
    file = open(record_name,'w')
    for file_index in range(1, 1+test_number):
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
        file.write('python train.py file_name=../1_year_data/{7}_train_{0}.csv model_record_path=../1_year_result/model_{1}/ file_record_path=../1_year_result/record_{2}/ method_name={3} scaler_name=scaler_{4}.m pca=False pca_name=pca_{5}.m model_name=model_{6}.m\n'.format(file_index, record_number, record_number, method_name, file_index, file_index, file_index, dataset_name))
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
        file.write('python test.py train_file_name=../1_year_data/{8}_train_{0}.csv test_file_name=../1_year_data/{8}_test_{0}.csv model_record_path=../1_year_result/model_{1}/ file_record_path=../1_year_result/record_{2}/ method_name={3} scaler_name=scaler_{4}.m model_name=model_{6}.m record_name=result_{7}.csv\n'.format(file_index, record_number, record_number, method_name, file_index, file_index, file_index, file_index, dataset_name, file_index))
    file.close()