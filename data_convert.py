
data_file = "new_datas.txt"
f_org = open(data_file, 'r')
#f_new = open("RARI_datas.txt", 'w')
f_new_tr = open("RARI_training.txt", 'w')
f_new_te = open("RARI_testing.txt", 'w')

i=0

while True:
    i = i+1
    line = f_org.readline()
    if not line: break
    line_s = line.split('\t')
    if float(line_s[5]) == 0.0 and float(line_s[6]) == 0.0 and int(line_s[7]) == 0:
        continue
    else:
#70% training, 30% testing
        if i%10 >=  3:   
            f_new_tr.write(line)
        elif i%10 < 3:
            f_new_te.write(line)


