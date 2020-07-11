import json
import argparse
import os
from ast import literal_eval
import numpy as np
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--json_path", required = True, help="Path to the json file")
args = ap.parse_args()
json_path = args.json_path

root = json_path[:json_path.rfind("/")]
label = ["leftleft", "left", "right", "rightright"]
points = []
ref_point = []


json_gt = [json.loads(line) for line in open(json_path)]
for gt in json_gt:
	lanes = gt['lanes']
	h_samples = gt['h_samples']
	img_file = gt['raw_file']
	raw_file = gt['raw_file'][:-4]
	path = raw_file[:raw_file.rfind("/")]
	number = raw_file[raw_file.rfind("/")+1:]
	#print(root+"/"+img_file)
	img = cv2.imread(root+"/"+img_file)
	width, height, channels = img.shape

	# take points in lanes
	for lane in lanes:
		for idx, point in enumerate(lane):
			if point != -2:
					ref_point.append((point, h_samples[idx]))
		points.append(ref_point)
		ref_point = []
	try:
		if not os.path.exists(path):
			os.makedirs(path)
	except OSError:
		print('Error: Creating directory of data')

	print(path)
	# make txt file about coordinates of points
	if len(points) <= 4:
		for save_num in range(int(number)):
			with open(root+"/"+path+"/"+str(save_num+1)+".txt", 'w') as output:
				for i, row in enumerate(points):
					output.write(label[i]+ " ")
					for data in row:
						output.write(str(data[0])+" "+str(data[1])+" ")
					output.write('\n')

	# make seg image
	seg_directory_path = root+"/laneseg/"+path
	if len(points) <= 4:
		try:
			if not os.path.exists(seg_directory_path):
				os.makedirs(seg_directory_path)
		except OSError:
			print('Error: Creating directory of data')
		seg = np.zeros((width, height,1), np.uint8)
		for i, lane in enumerate(points):
			for idx in range(len(lane)-1):
				seg = cv2.line(seg, tuple(lane[idx]), tuple(lane[idx+1]), (i+1), 16)

		for save_num in range(int(number)): 
			cv2.imwrite(seg_directory_path+"/"+str(save_num+1)+".png",seg)

	points=[]

print("Done!!!")
