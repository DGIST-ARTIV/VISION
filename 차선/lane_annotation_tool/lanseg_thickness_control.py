#import the necessary packages
import argparse
import os
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file_path", required = True, help="Path to the image file")
args = ap.parse_args()

path = args.file_path

none_lst = [10,6,7,12]

for root, dirs, files in os.walk(path):
    # root = root[root.find('/',3):]
    for fname in files:
        num = fname[:fname.find('.')]
        seg_path = root[:root.find("video")]+"laneseg_thick_control"+root[root.find("/",root.find("video")):]+"/"
        try:
            if not os.path.exists(seg_path):
                os.makedirs(seg_path)
        except OSError:
            print('Error: Creating directory of data')
        #print(num)
        if fname.endswith(".txt"):
            with open(root+"/"+fname, "rt") as output:
                seg = np.zeros((288,800,1), np.uint8)
                str_out = output.readlines()
                for idx, li in enumerate(str_out):
                    xy_list = li.split()[1:]
                    x = []
                    y = []
                    for i in range(int(len(xy_list)/2)):
                        x.append(int(xy_list[2*i]))
                        y.append(int(xy_list[2*i+1]))
                    for i in range(len(x)-1):
                        seg = cv2.line(seg, (x[i], y[i]), (x[i+1], y[i+1]), idx+1, 8)
                    print(seg_path+num+".png")
                    cv2.imwrite(seg_path+num+".png",seg)

