dataset_name = 'abalone19'
content = []

with open('./generate_execute.py', 'r+') as f:
    for line in f.readlines():
        if line.find('file_name_pre = ') != -1:
            content.append('file_name_pre = \'{0}\'\n'.format(dataset_name))
        else:
            content.append(line)
    f.seek(0)
    f.truncate()

    for line in content:
        f.write(line)