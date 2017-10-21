import sys
data_file = sys.argv[1]
out_file = sys.argv[2]

with open(data_file, 'r') as fin, open(out_file, 'w') as fout:
    for line in fin:
        ss = line.strip().split()
        fout.write(' '.join(ss[1:7]) + '\n')
