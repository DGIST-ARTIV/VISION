from os import listdir
from os.path import isfile, join
import os
path_0 = os.getcwd() + "/"
path = path_0 + "train2017"
files = [f for f in listdir(path) if isfile(join(path, f))]
#files = [x for x in files if x.find("t") != -1]
#print(files)
files = sorted(files)
f = open(path_0 + "label_list.txt",'a')
for i in range(len(files)):
    data = path + "/"+files[i]
    print(data)
    f.write(data)
    f.write("\n")
f.close()
