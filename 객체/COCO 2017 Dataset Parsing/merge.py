from os import listdir
from os.path import isfile, join
import os

path_0 = os.getcwd() + "/"
result_directory = "merge_person+car"
destination = path_0 + result_directory
path_person = path_0 + "person_labels"
path_car = path_0 + "car_labels"

files_person = [f for f in listdir(path_person) if isfile(join(path_person, f))]
files_car = [f for f in listdir(path_car) if isfile(join(path_car, f))]
#files = [x for x in files if x.find("t") != -1]

files_person = sorted(files_person)
files_car = sorted(files_car)

try:
	if not os.path.exists(destination):
	       os.makedirs(destination)
except OSError:
    print('Error: Creating %s directory' % result_directory)

for i in files_person:
	if i in files_car:
		print(i)
		f_person = open(path_person + "/" + i,'r')
		f_car = open(path_car + "/" + i,'r')
		f_new = open(destination + "/" + i,'w')
		line_p = f_person.readline()
		while line_p:
			f_new.write(line_p)
			print(line_p)
			line_p = f_person.readline()
		line_c = f_car.readline()
		while line_c:
			f_new.write(line_c)
			print(line_c)
			line_c = f_car.readline()
		f_person.close()
		f_car.close()
		f_new.close()
	else:
		print(i)
		f_person = open(path_person + "/" + i,'r')
		f_new = open(destination + "/" + i,'w')
		line_p = f_person.readline()
		while line_p:
			f_new.write(line_p)
			print(line_p)
			line_p = f_person.readline()
		f_person.close()
		f_new.close()

for i in files_car:
	if i in files_person:
		pass
	else:
		f_car = open(path_car + "/" + i,'r')
		f_new = open(destination + "/" + i,'w')
		line_c = f_car.readline()
		while line_c:
			f_new.write(line_c)
			print(line_c)
			line_c = f_car.readline()
		f_car.close()
		f_new.close()
