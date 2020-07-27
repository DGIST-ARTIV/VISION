from os import listdir
from os.path import isfile, join
import os
path = os.getcwd() + "/listing"
print(path)
files = [f for f in listdir(path) if isfile(join(path, f))]
files = [x for x in files if x.find(".txt") != -1]
#print(files)
files = sorted(files)
print(files)
f_result = open(path + "/train.txt",'w')
for i in files:
    print("--------------------------------")
    f = open(path+"/"+i, 'r')
    line = f.readline()
    while(line):
        print(line)
        f_result.write(line)
        line = f.readline()
    f.close()
f_result.close()
