content = []

with open('./generate_execute.py', 'r+') as f:
    for line in f.readlines():
        if line.find('python train') != -1:
            content.append('#' + line)
        elif line.find('python test') != -1:
            content.append(line[1:])
        else:
            content.append(line)
    f.seek(0)
    f.truncate()

    for line in content:
        f.write(line)