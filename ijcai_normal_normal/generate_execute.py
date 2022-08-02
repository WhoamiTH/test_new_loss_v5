import sys
method_name = 'ijcai'
train_method = 'normal'
test_method = 'normal'
total_record_number = 4
test_number = 5
winner_number = 3
file_name_pre = 'abalone19'
for record_number in range(1, total_record_number+1):
    record_name = 'execute_train_{0}.sh'.format(record_number)
    file = open(record_name,'w')
    for i in range(1, 1+test_number):
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
        file.write('python train_{1}.py file_name=../1_year_data/{3}_train_{4}.csv model_record_path=../1_year_result/{0}_{1}_{2}/model_{5}/ file_record_path=../1_year_result/{0}_{1}_{2}/record_{5}/ method_name={0}_{1}_{2} scaler_name=scaler_{4}.m pca=False pca_name=pca_{4}.m model_name=model_{4}/my_model\n'.format(method_name, train_method, test_method, file_name_pre, i, record_number))
    file.close()
    record_name = 'execute_test_{0}.sh'.format(record_number)
    file = open(record_name,'w')
    for i in range(1, 1+test_number):
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
        file.write('python test_{2}.py train_file_name=../1_year_data/{3}_train_{4}.csv test_file_name=../1_year_data/{3}_test_{4}.csv model_record_path=../1_year_result/{0}_{1}_{2}/model_{5}/ file_record_path=../1_year_result/{0}_{1}_{2}/record_{5}/ method_name={0}_{1}_{2} scaler_name=scaler_{4}.m model_name=model_{4} record_name=test_result_{4}.csv\n'.format(method_name, train_method, test_method, file_name_pre, i, record_number))
        """
        analyse parameter list:
        predict_label_file_name
        true_label_file_name
        model_record_path
        file_record_path
        method_name
        winner_number
        """
        # file.write('python analyse_result.py predict_label_file_name={0}_result_{1}.csv true_label_file_name={8}_test_origin_{2}.csv model_record_path=../1_year_result/model_{3}/ file_record_path=../1_year_result/record_{4}/ method_name={5} winner_number={6} record_name=result_{7}.txt'.format(method_name, i, i, record_number, record_number, method_name, winner_number, i, file_name_pre))
    file.close()
