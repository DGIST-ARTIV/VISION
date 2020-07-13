import rospy
import message_filters
from message_filters import ApproximateTimeSynchronizer, Subscriber
import cv2
import numpy as np
import os
import sys
import glob
import argparse
import torch
import time
import networks
from torchvision import transforms, datasets
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from YOLO.msg import BoundingBox, BoundingBoxes
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError

# Instantiate CvBridge
bridge = CvBridge()
classes_colors = {}
DepthMapPublisher = rospy.Publisher('YOLO_RESULT/DepthMap', Image, queue_size=1)
BboxImagePublisher = rospy.Publisher('YOLO_RESULT/DetectedImg', Image, queue_size=1)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use',
                        choices=[
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320"])
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument('--width', type=int,
                        help='desired width', default=1080)
    parser.add_argument('--height', type=int,
                        help='image extension to search for in folder', default=720)
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')

    return parser.parse_args()

def callback(data, msg):
    print("fuck")
    global FinalDepthMap
    global args
    global feed_width
    global feed_height
    global device
    global encoder
    global depth_decoder

    imgIn = bridge.imgmsg_to_cv2(msg, "bgr8")
    imgIn = cv2.cvtColor(imgIn, cv2.COLOR_RGB2BGR)
    height, width, channel = imgIn.shape
    imgforDepth = imgIn.copy()
    imgforBbox = imgIn.copy()
# =================================================================================
    with torch.no_grad():
        img = bridge.imgmsg_to_cv2(msg, "bgr8")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #img = cv2.resize(img, dsize=(width, height))
        original_img = img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = pil.fromarray(img)
        # Load image and preprocess
        input_image = im_pil
        original_width, original_height = input_image.size
        input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

        # *****Depth*****
        # PREDICTION
        input_image = input_image.to(device)
        features = encoder(input_image)
        outputs = depth_decoder(features)

        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(
        disp, (original_height, original_width), mode="bilinear", align_corners=False)

        # Saving colormapped depth image
        disp_resized_np = disp_resized.squeeze().cpu().numpy()
        print("disp_resized_np", disp_resized_np.max())
        vmax = np.percentile(disp_resized_np, 90)
        print("vmax", vmax)
        print("==========================================================================")

        normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
        #mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        mapper = cm.ScalarMappable(norm=None, cmap='magma')
        colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
        im = pil.fromarray(colormapped_im)
        open_cv_image = np.array(im)
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        FinalDepthMap = open_cv_image

        #fuck
        """
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
            crop = FinalDepthMap[xmin:xmax, ymin:ymax]
            cv2.rectangle(FinalDepthMap, (int(xmin)-5, int(ymin)-45), (int(xmax)+5, int(ymin)), (255,255,255),-1)
            cv2.rectangle(FinalDepthMap, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (color[0],color[1],color[2]),2)
            #print(color)
            text = ('{:s}: {:.3f}').format(label,confidence)
            cv2.putText(FinalDepthMap, text, (int(xmin), int(ymin-10)), font, fontScale, (0,0,0), thickness ,cv2.LINE_AA)
            cv2.putText(FinalDepthMap, line, (int(xmin), int(ymin-25)), font, fontScale, (0,0,0), thickness ,cv2.LINE_AA)
            """
#fuck
        print (open_cv_image.shape)
        #open_cv_image = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        image_msg_depth = bridge.cv2_to_imgmsg(FinalDepthMap, "rgb8")
        DepthMapPublisher.publish(image_msg_depth)

# =================================================================================
    # *****Bbox + Distance*****
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
        crop = cv2.cvtColor(FinalDepthMap, cv2.COLOR_BGR2GRAY)
        crop = FinalDepthMap[xmin:xmax, ymin:ymax]
        crop = np.array(crop)
        distance = str(round(crop.mean(),2))
        cv2.rectangle(imgforBbox, (int(xmin)-5, int(ymin)-65), (int(xmax)+5, int(ymin)), (255,255,255),-1)
        cv2.rectangle(imgforBbox, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (color[0],color[1],color[2]),2)
        #print(color)
        text = ('{:s}: {:.3f}').format(label,confidence)
        cv2.putText(imgforBbox, text, (int(xmin), int(ymin-10)), font, fontScale, (0,0,0), thickness ,cv2.LINE_AA)
        cv2.putText(imgforBbox, "which:"+line, (int(xmin), int(ymin-25)), font, fontScale, (0,0,0), thickness ,cv2.LINE_AA)
        cv2.putText(imgforBbox, "D:"+distance, (int(xmin), int(ymin-45)), font, fontScale, (0,0,0), thickness ,cv2.LINE_AA)

    image_msg_bbox = bridge.cv2_to_imgmsg(imgforBbox, "rgb8")
    BboxImagePublisher.publish(image_msg_bbox)

def main():
    global args
    global device
    global encoder
    global depth_decoder
    global encoder
    global feed_width
    global feed_height

    args = parse_args()
    width = args.width
    height = args.height
    """Function to predict for a single image or folder of images"""
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("models", args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("-> Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("-> Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))
    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)
    depth_decoder.to(device)
    depth_decoder.eval()
    #===================================================================================================
    # *****ROS*****
    rospy.init_node("PostProcessing")
    # Define your image topic
    # Set up your subscriber and define its callback
    bbox = message_filters.Subscriber("/yolov3/bbox", BoundingBoxes)
    image_raw = message_filters.Subscriber("/yolov3/image_raw", Image)
    #DepthMap = rospy.Subscriber("/yolov3/image_raw", Image, Depthcallback)
    ts = message_filters.ApproximateTimeSynchronizer([bbox, image_raw], 10, 10)
    ts.registerCallback(callback)
    rospy.spin()

if __name__ == '__main__':
    main()
