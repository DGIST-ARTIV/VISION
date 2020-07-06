import cv2
import numpy as np

a = cv2.imread("/home/vision/eunbin/annotation_tool/laneseg/video.webm_annot/1010.png")

a[np.where(a!=0)]=a[np.where(a!=0)]*50

cv2.imshow("a", a)

cv2.waitKey(0)
