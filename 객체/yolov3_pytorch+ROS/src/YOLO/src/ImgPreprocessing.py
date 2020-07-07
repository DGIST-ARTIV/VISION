#! /usr/bin/python
# Copyright (c) 2015, Rethink Robotics, Inc.

# Using this CvBridge Tutorial for converting
# ROS images to OpenCV2 images
# http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython

# Using this OpenCV2 tutorial for saving Images:
# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html

# rospy for the subscriber
import rospy
import message_filters
import cv2
import numpy as np
# ROS Image message
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image

from std_msgs.msg import String
from YOLO.msg import BoundingBox, BoundingBoxes

# Instantiate CvBridge
bridge = CvBridge()

global num
global xmin
global xmax
global ymax
global width
global height
global center_x
global center_y

bridge = CvBridge()
classes_colors = {}
pub = rospy.Publisher('YOLO_RESULT', Image, queue_size=1)

def callback(data, msg):
    # Convert your ROS Image message to OpenCV2
    imgIn = bridge.imgmsg_to_cv2(msg, "bgr8")
    imgIn = cv2.cvtColor(imgIn, cv2.COLOR_RGB2BGR)
    height, width, channel = imgIn.shape
    print width
    imgOut = imgIn.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    thickness = 2
    for index in range(len(data.bounding_boxes)):
        label = data.bounding_boxes[index].Class
        xmin = data.bounding_boxes[index].xmin
        ymin = data.bounding_boxes[index].ymin
        xmax = data.bounding_boxes[index].xmax
        ymax = data.bounding_boxes[index].ymax
        confidence = data.bounding_boxes[index].probability
        w = int(xmax - xmin)
        h = int(ymax - ymin)
        center_x = int(xmin + w/2)
        center_y = int(ymin + h/2)

    # Find class color
        if label in classes_colors.keys():
            color = classes_colors[label]
        else:
            # Generate a new color if first time seen this label
            color = np.random.randint(0,188,3)
            #color[0] = (255,255,255)
            #color[1] = (198,198,198)
            #color[2] = (27,27,27)
            classes_colors[label] = color

        # line select
        if center_x < width//4:
            line = "L"
        elif width//4 < center_x < (width//4) * 3:
            line = "C"
        else:
            line = "R"

        # Create rectangle
        cv2.rectangle(imgOut, (int(xmin)-5, int(ymin)-45), (int(xmax)+5, int(ymin)), (255,255,255),-1)
        cv2.rectangle(imgOut, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (color[0],color[1],color[2]),2)
        #print(color)
        text = ('{:s}: {:.3f}').format(label,confidence)
        cv2.putText(imgOut, text, (int(xmin), int(ymin-10)), font, fontScale, (0,0,0), thickness ,cv2.LINE_AA)
        cv2.putText(imgOut, line, (int(xmin), int(ymin-25)), font, fontScale, (0,0,0), thickness ,cv2.LINE_AA)

    image_msg = bridge.cv2_to_imgmsg(imgOut, "rgb8")
    pub.publish(image_msg)

def main():
    rospy.init_node("DetectedImg")
    # Define your image topic
    # Set up your subscriber and define its callback
    bbox = message_filters.Subscriber("/yolov3/bbox", BoundingBoxes)
    image_raw = message_filters.Subscriber("/yolov3/image_raw", Image)
    ts = message_filters.ApproximateTimeSynchronizer([bbox, image_raw], 10, 0.03)
    ts.registerCallback(callback)
    rospy.spin()

if __name__ == '__main__':
    main()
