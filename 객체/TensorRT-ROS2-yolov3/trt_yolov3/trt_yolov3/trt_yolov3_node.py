import os
import cv2
import rclpy
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time
import argparse
import pycuda.autoinit  # This is needed for initializing CUDA driver

from yolo_bbox.msg import BoundingBox, BoundingBoxes
from trt_yolov3.yolov3 import TrtYOLOv3
from trt_yolov3.camera import add_camera_args, Camera
from trt_yolov3.display import open_window, set_display, show_fps
from trt_yolov3.visualization import BBoxVisualization

bridge = CvBridge()

COCO_CLASSES_LIST = [
    'person',
    'bicycle',
    'car',
    'motorbike',
    'aeroplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'sofa',
    'pottedplant',
    'bed',
    'diningtable',
    'toilet',
    'tvmonitor',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
]

def get_cls_dict(category_num):
    """Get the class ID to name translation dictionary."""
    if category_num == 80:
        return {i: n for i, n in enumerate(COCO_CLASSES_LIST)}
    else:
        return {i: 'CLS%d' % i for i in range(category_num)}


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
    print("image_callback called")
    print("==============================")
    print("======= imgmsg to cv2 ==========")
    time_now = time.time()
    img = bridge.imgmsg_to_cv2(msg, "rgb8")
    print(time.time() - time_now)
    print("==============================")

    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    print("======= detect ==========")
    time_now = time.time()
    boxes, confs, clss = trt_yolov3.detect(img, conf_th)
    print(time.time() - time_now)
    print("==============================")
    print("======= message mapping ==========")
    time_now = time.time()
    boxes = boxes.tolist()
    confs = confs.tolist()
    clss = clss.tolist()
    detection_results = BoundingBoxes()
    detection_results.header = msg.header
    detection_results.image_header = msg.header

    if boxes is not None:
        for i in range(len(boxes)):
            # Populate darknet message
            detection_msg = BoundingBox()
            detection_msg.xmin = boxes[i][0]
            detection_msg.xmax = boxes[i][1]
            detection_msg.ymin =  boxes[i][2]
            detection_msg.ymax = boxes[i][3]
            detection_msg.probability = confs[i]
            detection_msg.class_type = str(get_cls_dict(clss[i]))
            # Append in overall detection message
            detection_results.bounding_boxes.append(detection_msg)
    detection_results.header.stamp = node.get_clock().now().to_msg()
    pub_.publish(detection_results)
    print(time.time() - time_now)
    print("==============================")
    print("======= cv2 to imgmsg ==========")
    # Publish detection results
    img = vis.draw_bboxes(img, boxes, confs, clss)
    time_now = time.time()
    imgmsg = bridge.cv2_to_imgmsg(img, "rgb8")
    pub.publish(imgmsg)
    print(time.time()-time_now)
    print("==========================")

node = 0

def main():
    global trt_yolov3
    global conf_th
    global vis
    global pub
    global pub_
    global node
    conf_th = 0.5
    rclpy.init(args=None)
    args = parse_args()
    node = rclpy.create_node('image_sub_py')
    sub = node.create_subscription(Image, '/usb_cam/image_raw', image_callback)
    pub = node.create_publisher(Image, '/TRT_yolov3/result_image')
    pub_ = node.create_publisher(BoundingBoxes, '/TRT_yolov3/Bbox')

    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    #if not os.path.isfile('.%s.trt' % args.model):
    #    raise SystemExit('ERROR: file (/%s.trt) not found!' % args.model)

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
