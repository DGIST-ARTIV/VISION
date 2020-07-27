from os import listdir
from os.path import isfile, join
import os
import cv2

path_0 = os.getcwd() + "/"
path_label = path_0 + "car_labels"
path_train = path_0 + "train2017"
files_label = [f[:-4] for f in listdir(path_label) if isfile(join(path_label, f))]
files_train = [f[:-4] for f in listdir(path_train) if isfile(join(path_train, f))]
#files = [x for x in files if x.find("t") != -1]

files_label = sorted(files_label)
files_train = sorted(files_train)

try:
	if not os.path.exists(path_0 + "car_image"):
	       os.makedirs(path_0 + "car_image")
except OSError:
    print('Error: Creating directory of data')

for i in files_label:
    for j in files_train:
        if i == j:
            print("i: ", i)
            print("j: ", type(j))
            img = cv2.imread(path_train  + "/" + j + ".jpg")
            cv2.imwrite("./car_image/" + j + ".jpg", img)
            print("now, saving" + i + ".jpg")

        else:
            pass
            #print(i + ".jpg" + " is not exist.")


print("Finish!")
