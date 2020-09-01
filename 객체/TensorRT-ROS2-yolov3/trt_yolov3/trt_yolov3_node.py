### Update List
# Date: 0807
# Author: Gu Lee
# No more custom messages are used. Now use  Int16MultiArray instead of BoundingBox and BoundingBoxes me
# Through this, TRT_yolov3/Bbox topic can be recorded with rosbag.

# Date: 0816
# Author: Gu Lee
# Separate image callback and detection function.
# Use rclpy.spin_once instead of rclpy.spin


import os
import cv2
import rclpy
import numpy as np
from std_msgs.msg import String, Int16MultiArray, MultiArrayDimension
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time
import argparse
import pycuda.autoinit  # This is needed for initializing CUDA driver
from trt_yolov3.yolov3 import TrtYOLOv3
from trt_yolov3.camera import add_camera_args, Camera
from trt_yolov3.display import open_window, set_display, show_fps
from trt_yolov3.visualization import BBoxVisualization

bridge = CvBridge()

COCO_CLASSES_LIST = ['person','bicycle','car','motorbike','aeroplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parkingmeter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis',
'snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork',
'knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','sofa','pottedplant',
'bed','diningtable','toilet','tvmonitor','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator',
'book','clock','vase','scissors','teddy bear','hair drier','toothbrush',]

KCITY_CUSTOM = ['car','crossWalk_sign','bicycle_sign','bust_sign','construction_sign','parking_sign','kidSafeSero_sign','busArrowDown_sign','trafficLightRedYellow','trafficLightGreenLeft',
'trafficLightYellow','trafficGreen','trafficLightRed','trafficLightRedLeft',]

def get_cls_dict(category_num):
    """Get the class ID to name translation dictionary."""
    if category_num == 80:
        return {i: n for i, n in enumerate(COCO_CLASSES_LIST)}
    elif category_num == 14:
        return {i: n for i, n in enumerate(KCITY_CUSTOM)}

def parse_args():
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLOv3 model on Jetson Nano')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument('--model', type=str, default='yolov3-416',
                        help='yolov3[-spp|-tiny]-[288|416|608]')
    parser.add_argument('--category_num', type=int, default=80,
                        help='number of object categories [80]')
    args = parser.parse_args()

    os.system("ls")

    return args

def image_callback(msg : Image):
    global trt_yolov3
    global conf_th
    global vis
    global pub
    global pub_
    global node
    global img
    now = time.time()
    time_now = time.time()
    img = bridge.imgmsg_to_cv2(msg, "rgb8")
    boxes, confs, clss = trt_yolov3.detect(img, conf_th)
    boxes = boxes.tolist()
    confs = confs.tolist()
    clss = clss.tolist()
    detection_results = Int16MultiArray()
    if boxes is not None:
        for i in range(len(boxes)):
                detection_results.layout.dim.append(MultiArrayDimension())
                detection_results.layout.dim[i].label = "object" + str(i)
                detection_results.layout.dim[i].size = 6
                detection_results.data = [0] * 6 * len(boxes)
                detection_results.data[i * 6] = clss[i]
                detection_results.data[i * 6 + 1] = int(confs[i] * 100)
                detection_results.data[i * 6 + 2] = boxes[i][0]
                detection_results.data[i * 6 + 3] = boxes[i][1]
                detection_results.data[i * 6 + 4] = boxes[i][2]
                detection_results.data[i * 6 + 5] = boxes[i][3]
    pub_.publish(detection_results)

    # for result image publish
    # This leads to a serious decline in performance.
    # Use just for Debugging    
    img = vis.draw_bboxes(img, boxes, confs, clss)
    img = cv2.resize(img, None, fx=0.2, fy=0.2)
    imgmsg = bridge.cv2_to_imgmsg(img, "rgb8")
    pub.publish(imgmsg)

node = 0

def main():
    global trt_yolov3
    global conf_th
    global vis
    # for result image publish
    global pub
    global pub_
    global node
    conf_th = 0.5
    rclpy.init(args=None)
    args = parse_args()
    node = rclpy.create_node('image_sub_py')
    #sub = node.create_subscription(Image, '/image', image_callback)
    sub = node.create_subscription(Image, '/movie', image_callback)
    
    # for result image publish    
    pub = node.create_publisher(Image, '/TRT_yolov3/result_image')
    pub_ = node.create_publisher(Int16MultiArray, '/TRT_yolov3/Bbox')

    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
 
    cls_dict = get_cls_dict(args.category_num)
    yolo_dim = args.model.split('-')[-1]
    if 'x' in yolo_dim:
        dim_split = yolo_dim.split('x')
        if len(dim_split) != 2:
            raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)
        w, h = int(dim_split[0]), int(dim_split[1])
    else:
        h = w = int(yolo_dim)
    if h % 32 != 0 or w % 32 != 0:
        raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)
    trt_yolov3 = TrtYOLOv3(args.model, (h, w), args.category_num)
    vis = BBoxVisualization(cls_dict)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
