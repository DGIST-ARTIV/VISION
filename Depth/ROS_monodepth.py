from __future__ import absolute_import, division, print_function
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import cv2
import torch
from torchvision import transforms, datasets
import time
import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
from std_msgs.msg import String
from YOLO.msg import BoundingBox, BoundingBoxes

def parse_args():
    global parser
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
                        help='desired height', default=720)
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')

    return parser.parse_args()

# Instantiate CvBridge
bridge = CvBridge()

def callback(msg):
    global feed_width
    global feed_height
    global device
    global encoder
    # Convert your ROS Image message to OpenCV2
    #print "callback called"
    img = bridge.imgmsg_to_cv2(msg, "bgr8")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #cv2 image
    print(time.time())
    img = cv2.resize(img, dsize=(1080, 720))
    original_img = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = pil.fromarray(img)
    # Load image and preprocess
    input_image = im_pil
    original_width, original_height = input_image.size
    input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)

    # PREDICTION
    input_image = input_image.to(device)
    features = encoder(input_image)
    outputs = depth_decoder(features)

    disp = outputs[("disp", 0)]
    disp_resized = torch.nn.functional.interpolate(
    disp, (original_height, original_width), mode="bilinear", align_corners=False)

    # Saving colormapped depth image
    disp_resized_np = disp_resized.squeeze().cpu().numpy()

    vmax = np.percentile(disp_resized_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='gray')
    colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
    im = pil.fromarray(colormapped_im)
    open_cv_image = np.array(im)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    image_msg = bridge.cv2_to_imgmsg(open_cv_image, "rgb8")
    pub.publish(image_msg)

def main():
    global device
    args = parse_args()
    rospy.init_node("DepthMap")
    #print "start"
    # Define your image topic
    # Set up your subscriber and define its callback
    rospy.Subscriber('/usb_cam/image_raw', Image, callback)

    global parser
    width = args.width
    height = args.height
    """Function to predict for a single image or folder of images
    """

    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("models", args.model_name)
    #print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    #print("   Loading pretrained encoder")
    global encoder
    encoder = networks.ResnetEncoder(18, False)
    global loaded_dict_enc
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    global feed_width
    global feed_height
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}

    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()


    #print("   Loading pretrained decoder")
    global depth_decoder
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)
    depth_decoder.to(device)
    depth_decoder.eval()
    print("fuck")

    rospy.spin()

if __name__ == '__main__':
    main()
