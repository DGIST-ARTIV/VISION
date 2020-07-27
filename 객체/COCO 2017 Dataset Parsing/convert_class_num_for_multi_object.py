# -*- coding: utf-8 -*-
import os
from os import listdir
from os.path import isfile, join
import codecs

desired_class_num = input("write desired class num!\n")
path_0 = os.getcwd() + "/"
files = [f for f in listdir(path_0) if isfile(join(path_0, f))]
files = [x for x in files if x.find(".txt") != -1]

files = sorted(files)
for i in files:
    lines = ""
    with codecs.open(path_0 + i, 'r', encoding='utf-8',errors='ignore') as f:
        line = f.readline()
        while line:
            line = line.split()
            line[0] = desired_class_num
            line = " ".join(line)
            lines += line
            line = f.readline()        
            lines += "\n"

    with codecs.open(path_0 + i, 'w', encoding='utf-8',errors='ignore') as f:
        f.write(lines)
        
print("Finish!")
