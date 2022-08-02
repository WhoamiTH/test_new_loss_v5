content = []

import glob


path = './*'
file_list = glob.glob(path)
# file_list.sort()

for i in file_list:
    if i.find('test_') != -1:
        test_file_name = i

with open(test_file_name, 'r+') as f:
    for line in f.readlines():
        if line.find('os.environ') != -1:
            content.append('#' + line)
        else:
            content.append(line)
        # elif line.find('python test') != -1:
        #     content.append(line[1:])
        # else:
        #     content.append(line)
    f.seek(0)
    f.truncate()

    for line in content:
        f.write(line)