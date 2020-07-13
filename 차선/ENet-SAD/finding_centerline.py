import argparse
import cv2
import torch
torch.cuda.set_device(1)
from model import SCNN
from model_ENET_SAD import ENet_SAD
from utils.prob2lines import getLane
from utils.transforms import *
import numpy as np

import time
from multiprocessing import Process, JoinableQueue, SimpleQueue
from threading import Lock

from numba import jit

b = (255,0,0)
g = (0,255,0)
r = (0,0,255)
p = (196,37,244)
colorlist=[r,g,b,p]

img_size = (800, 288)
#net = SCNN(input_size=(800, 288), pretrained=False)
net = ENet_SAD(img_size, sad=False)
# CULane mean, std
mean=(0.3598, 0.3653, 0.3662)
std=(0.2573, 0.2663, 0.2756)
# Imagenet mean, std
# mean=(0.485, 0.456, 0.406)
# std=(0.229, 0.224, 0.225)
transform_img = Resize(img_size)
transform_to_net = Compose(ToTensor(), Normalize(mean=mean, std=std))

pipeline = False

prepoints = 1 # this for count lane...
def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--video_path", '-i', type=str, default="demo/ioniq.webm", help="Path to demo video")
    parser.add_argument("--video_path", '-i', type=str, default="/home/dgist/Desktop/data/원내주행영상/낮/0626/E16todorm.webm", help="Path to demo video")
    #parser.add_argument("--video_path", '-i', type=str, default="/home/dgist/Desktop/data/원내주행영상/낮/0626/long.webm", help="Path to demo video")
    parser.add_argument("--weight_path", '-w', type=str, default="experiments/exp1_culane/exp1_best.pth", help="Path to model weights")
    parser.add_argument("--visualize", '-v', action="store_true", default=True, help="Visualize the result")
    args = parser.parse_args()
    return args


def network(net, img):
    seg_pred, exist_pred = net(img.cuda())[:2]
    seg_pred = seg_pred.detach().cpu()
    exist_pred = exist_pred.detach().cpu()
    return seg_pred, exist_pred


def visualize(img, seg_pred, exist_pred):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    lane_img = np.zeros_like(img)
    color = np.array([[255, 125, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255]], dtype='uint8')
    coord_mask = np.argmax(seg_pred, axis=0)
    for i in range(0, 4):
        if exist_pred[0, i] > 0.5:
            lane_img[coord_mask == (i + 1)] = color[i]
    img = cv2.addWeighted(src1=lane_img, alpha=0.8, src2=img, beta=1., gamma=0.)
    return img


def pre_processor(arg):
    img_queue, video_path = arg
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        if img_queue.empty():
            ret, frame = cap.read()
            if ret:
                #frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                frame = transform_img({'img': frame})['img']
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                x = transform_to_net({'img': img})['img']
                x.unsqueeze_(0)

                img_queue.put(x)
                img_queue.join()
            else:
                break


def post_processor(arg):
    img_queue, arg_visualize = arg

    while True:
        if not img_queue.empty():
            x, seg_pred, exist_pred = img_queue.get()
            seg_pred = seg_pred.numpy()[0]
            exist_pred = exist_pred.numpy()

            exist = [1 if exist_pred[0, i] > 0.5 else 0 for i in range(4)]

            for i in getLane.prob2lines_CULane(seg_pred, exist):
            #for i in getLane.getLanePoints(seg_pred, exist):
                print(i)

            if arg_visualize:
                frame = x.squeeze().permute(1, 2, 0).numpy()
                img = visualize(frame, seg_pred, exist_pred)
                img = draw_regression_lane(i, img)
                cv2.imshow('input_video', frame)
                cv2.imshow("output_video", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            pass

def ransac_polyfit(x, y, img, order=2, n=3, k=10, t=5, d=4, f=0):
  # n – minimum number of data points required to fit the model
  # k – maximum number of iterations allowed in the algorithm
  # t – threshold value to determine when a data point fits a model
  # d – number of close data points required to assert that a model fits well to data
  # f – fraction of close data points required
  
  besterr = np.inf
  bestfit = None
  for_if = None
  for kk in range(k):
    try:
      maybeinliers = np.random.randint(len(x), size=n)
      maybemodel = np.polyfit(x[maybeinliers], y[maybeinliers], order)
      alsoinliers = np.abs(np.polyval(maybemodel, x)-y) < t
      print(np.abs(np.polyval(maybemodel, x)-y))
      if sum(alsoinliers) > d and sum(alsoinliers) > len(x)*f:
        bettermodel = np.polyfit(x[alsoinliers], y[alsoinliers], order)
        thiserr = np.sum(np.abs(np.polyval(bettermodel, x[alsoinliers])-y[alsoinliers]))
        if thiserr < besterr:
          bestfit = bettermodel
          besterr = thiserr
          for_if ="done"
    except:
      pass
  if for_if != None :
    draw_poly = np.array([list(zip(x[alsoinliers], np.polyval(bestfit, x[alsoinliers])))], np.int32)
    #print("draw_poly: ", np.polyval(bestfit, x[alsoinliers]))
    img = cv2.polylines(img, [draw_poly], False, (255,255,255),5)
  return img


def draw_lane_simple_method(line, img):
  for pts in range(len(line)):
    if line != [None] and pts != len(line)-1:
      img = cv2.line(img,tuple(line[pts]),tuple(line[pts+1]),(pts*50,0,0),2)
  return img

def draw_linear_regression_lane(line, img):
  global count, colorlist
  lane_img = np.zeros_like(img)  
  points =[]
  # (0, 220)   (130, 286)   (600,287)   (745, 236)
  x_list = []
  y_list = []
  if line != [None]:
    for idx, pts in enumerate(line):
      if 50<int(pts[0])<750 and 160 < int(pts[1]) < 260:
        x, y = int(pts[0]), int(pts[1])
        points.append((x,y))
        x_list.append(x)
        y_list.append(y)
  # linear regression
  vx, vy, x, y = cv2.fitLine(np.array(points), cv2.DIST_L2, 0, 0.11, 0.11)
  line = [float(vx),float(vy),float(x),float(y)]
  left_pt = int((-x*vy/vx) + y)
  right_pt = int(((lane_img.shape[1]-x)*vy/vx)+y)
  lane_img = cv2.line(lane_img,(lane_img.shape[1]-1,right_pt),(0,left_pt),255,2)
  img = cv2.addWeighted(src1=lane_img, alpha=1, src2=img, beta=1., gamma=0.)
  return img

def draw_polynomial_regression_lane(line, img):
  global count, colorlist
  lane_img = np.zeros_like(img)  
  points =[]
  # (0, 220)   (130, 286)   (600,287)   (745, 236)
  x_list = []
  y_list = []
  if line != [None] :
    for idx, pts in enumerate(line):
      if 50<int(pts[0])<750 and 160 < int(pts[1]) < 260:
        x, y = int(pts[0]), int(pts[1])
        points.append((x,y))
        x_list.append(x)
        y_list.append(y)

  # polynomial regression
  fp1 = np.polyfit(np.array(x_list), np.array(y_list) , 2)
  f1 = np.poly1d(fp1)
  y_list = np.polyval(f1, x_list)
  draw_poly = np.array([list(zip(x_list, y_list))], np.int32)
  print(np.asarray(draw_poly))
  lane_img=cv2.polylines(lane_img, np.asarray(draw_poly), False, (255,255,255), 5)
  #lane_img = ransac_polyfit(np.array(x_list), np.array(y_list),lane_img)
  img = cv2.addWeighted(src1=lane_img, alpha=1, src2=img, beta=1., gamma=0.)
  return img

def draw_lane_ransac(line, img):
  # global count, colorlist
  lane_img = np.zeros_like(img)  
  points =[]
  # (0, 220)   (130, 286)   (600,287)   (745, 236)
  x_list = []
  y_list = []
  for idx, pts in enumerate(line):
    if line != [None] and 50<int(pts[0])<750 and 160 < int(pts[1]) < 260:
      x, y = int(pts[0]), int(pts[1])
      #cv2.circle(img, (x,y), 10, colorlist[count-1], -1)
      #cv2.putText(img, str(idx), (x, y),cv2.FONT_HERSHEY_COMPLEX,0.4,(255,255,255),1)
      points.append((x,y))
      x_list.append(x)
      y_list.append(y)

  lane_img = ransac_polyfit(np.array(x_list), np.array(y_list),lane_img)
  img = cv2.addWeighted(src1=lane_img, alpha=1, src2=img, beta=1., gamma=0.)
  #count +=1
  return img

@jit(nopython=True, cache=True)
def for_degree(x1,x2,y1,y2):
    return (np.arctan2(x1-x2, y1-y2) * 180) / np.pi

def using_degree(line, img, start_point, idx):
  degree = [[-85, -75], [-60, -50], [45, 55], [70, 80]]
  #print(degree[idx][1])
  x1, y1 = start_point[0], start_point[1]
  x_new = [start_point[0]]
  y_new = [start_point[1]]
  #print("::::pts::::",line)
  for pts in line:
    if 50 <int(pts[0])<750 and 160 < int(pts[1]) < 280:
      x2, y2 = int(pts[0]), int(pts[1])
      # cv2.circle(img, (x2,y2), 10, colorlist[count-1], -1)
      # cv2.putText(img, str(idx), (x2, y2),cv2.FONT_HERSHEY_COMPLEX,0.4,(255,255,255),1)
      #slope_degree = for_degree(x1,x2, y1, y2)
      slope_degree = (np.arctan2(x1-x2, y1-y2) * 180) / np.pi
      if degree[idx][0] < (slope_degree) < degree[idx][1]:
        x_new.append(x2)
        y_new.append(y2)
  return x_new, y_new

def draw_lanes(lanes, img):
  lane_img = np.zeros_like(img)  
  start_points = [[0, 220], [180, 286], [580,287], [798, 240]]
  for idx, line in enumerate(lanes):
    if line == [None]:
      pass
    else:
      x_list, y_list = using_degree(line, img, start_points[idx], idx)
      # print("::::::new_list:::::",x_list, y_list)
      try:
        fp1 = np.polyfit(np.array(x_list), np.array(y_list) , 2)
        f1 = np.poly1d(fp1)
        # print("FFF!", f1)
        y_range = list(range(0, 280, 10))
        y_list = np.polyval(f1, x_list)
        draw_poly = np.array([list(zip(x_list, y_list))], np.int32)
        # print(np.asarray(draw_poly))
        lane_img=cv2.polylines(lane_img, np.asarray(draw_poly), False, (255,255,255),5)
        img = cv2.addWeighted(src1=lane_img, alpha=1, src2=img, beta=1., gamma=0.)
      except:
        pass
  return lane_img, img

def draw_lanes_ver2(lanes, img):
  lane_img = np.zeros_like(img)  
  start_points = [[0, 220], [180, 286], [580,287], [798, 240]]
  for idx, line in enumerate(lanes):
    if line == [None]:
      pass
    else:
      x_list, y_list = using_degree(line, img, start_points[idx], idx)
      # print("::::::new_list:::::",x_list, y_list)
      try:
        fp1 = np.polyfit(np.array(x_list), np.array(y_list) , 2)
        f1 = np.poly1d(fp1)
        print("FFF!", f1)
        y_range = list(range(x_list[:-1], 800, 10))
        y_list = np.polyval(f1, x_list)
        draw_poly = np.array([list(zip(x_list, y_list))], np.int32)
        # print(np.asarray(draw_poly))
        lane_img=cv2.polylines(lane_img, np.asarray(draw_poly), False, (255,255,255),5)
        img = cv2.addWeighted(src1=lane_img, alpha=1, src2=img, beta=1., gamma=0.)
      except:
        pass
  return lane_img, img

def affine_trasform(lane_img):
    # coordinate lu -> ld -> ru -> rd
    pts1 = np.float32([[265,165],[0, 215],[500,165],[798, 240]])

    # pts2 is points to move from pts1.
    pts2 = np.float32([[0,0],[0,300],[300,0],[300,300]])
    M = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.warpPerspective(lane_img, M, (300,300))
    dst = cv2.resize(dst, dsize=(70,70))
    return dst

def overwrap(lane_img, img):
  rows, cols, channels = lane_img.shape
  roi = img[10:rows+10, 10:cols+10]

  img2gray = cv2.cvtColor(lane_img, cv2.COLOR_BGR2GRAY)
  ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
  mask_inv = cv2.bitwise_not(mask)

  img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

  img2_fg = cv2.bitwise_and(lane_img, lane_img, mask=mask)

  dst = cv2.add(img1_bg, img2_fg)
  img[10:rows+10, 10:cols+10] = dst

  return img

# when change in the number of lanes, no recognition, not enough sample points, make re-evalution sign
def re_eval_sign(points):
    global prepoints
    if prepoints == 0:
        return False
    counter = 0
    for i in range(4):
        if points[i] != [None]:
            counter +=1
    if counter != prepoints:
        prepoints = counter
        return True
    prepoints = counter
    return False
        

def centerline_visualize(img, seg_pred, exist_pred):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # cv2.imshow("hls", img)
    mask_img = np.zeros((6,6))
    color = np.array([[255, 125, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255]], dtype='uint8')
    coord_mask = np.argmax(seg_pred, axis=0)
    color_choice = []
    # detect_white(img)
    # detect_yellow(img)
    try:
        for i in range(0, 4):
            voting_list = []
            if exist_pred[0][i] > 0.5:
                # mask_img[coord_mask == (i+1)] = img[coord_mask == (i+1)]
                for j in range(3):
                    idx = random.randint(0, len(np.where(coord_mask==(i+1))[0])-1)
                    x, y = np.where(coord_mask==(i+1))[0][idx], np.where(coord_mask==(i+1))[1][idx]
                    mask_img = img[x-3:x+3, y-3:y+3]
                    # cv2.imshow("mask_img", mask_img)
                    voting_list.append(decision(detect_white(mask_img), detect_yellow(mask_img)))
                    '''
                    if detect_yellow(mask_img) == "y":
                        voting_list.append("y")
                    elif detect_white(mask_img) == "w":
                        voting_list.append("w")
                    else:
                        voting_list.append("N")
                    '''
                color_choice.insert(i, voting(voting_list))
            else:
                color_choice.insert(i, "No_detection")
        print(color_choice)
    except:
        pass


def detect_white(mask_img):
    # White-ish areas in image
    # H value can be arbitrary, thus within [0 ... 360] (OpenCV: [0 ... 180])
    # L value must be relatively high (we want high brightness), e.g. within [0.7 ... 1.0] (OpenCV: [0 ... 255])
    # S value must be relatively low (we want low saturation), e.g. within [0.0 ... 0.3] (OpenCV: [0 ... 255])
    white_lower = np.array([np.round(  0 / 2), np.round(0.75 * 255), np.round(0.00 * 255)])
    white_upper = np.array([np.round(360 / 2), np.round(1.00 * 255), np.round(0.30 * 255)])
    try:
        white_mask = cv2.inRange(mask_img, white_lower, white_upper)

        return len(np.where(white_mask!= 0)[0])
    except:
        return 0

def detect_yellow(mask_img):
    # Yellow-ish areas in image
    # H value must be appropriate (see HSL color space), e.g. within [40 ... 60]
    # L value can be arbitrary (we want everything between bright and dark yellow), e.g. within [0.0 ... 1.0]
    # S value must be above some threshold (we want at least some saturation), e.g. within [0.35 ... 1.0]
    yellow_lower = np.array([np.round( 30 / 2), np.round(0.00 * 255), np.round(0.35 * 255)])
    yellow_upper = np.array([np.round( 60 / 2), np.round(1.00 * 255), np.round(1.00 * 255)])
    try:
        yellow_mask = cv2.inRange(mask_img, yellow_lower, yellow_upper)
        return len(np.where(yellow_mask!= 0)[0])
    except:
        return 0

    '''
        if len(np.where(yellow_mask!= 0)[0]) !=0:
            return "y"
        else:
            return "N"
    except:
        return "N"
    '''

def decision(white, yellow):
    if white > yellow:
        return "w"
    elif white <= yellow:
        if yellow !=0:
            return "y"
        return "N"
    else:
        return "N"


def voting(voting_list):
    if voting_list.count("y") != 0:
        if voting_list.count("w") ==2:
            return "w"
        return "y"
    elif voting_list.count("w") != 0:
        return "w"
    else:
        return "N"


def main():
    args = parse_args()
    video_path = args.video_path
    weight_path = args.weight_path
    global prepoints
    
    if pipeline:
        input_queue = JoinableQueue()
        pre_process = Process(target=pre_processor, args=((input_queue, video_path),))
        pre_process.start()

        output_queue = SimpleQueue()
        post_process = Process(target=post_processor, args=((output_queue, args.visualize),))
        post_process.start()
    else:
        cap = cv2.VideoCapture(video_path)

    save_dict = torch.load(weight_path, map_location='cpu')
    #save_dict['net']['fc.0.weight'] = save_dict['net']['fc.0.weight'].view(128,4400)
    #print(save_dict['net']['fc.0.weight'].view(128,4400))
    net.load_state_dict(save_dict['net'])

    net.eval()
    net.cuda()

    while True:
        if pipeline:
            loop_start = time.time()
            x = input_queue.get()
            input_queue.task_done()

            gpu_start = time.time()
            seg_pred, exist_pred = network(net, x)
            gpu_end = time.time()

            output_queue.put((x, seg_pred, exist_pred))

            loop_end = time.time()

        else:
            if not cap.isOpened():
                break

            ret, frame = cap.read()

            if ret:
                loop_start = time.time()
                #frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                frame = transform_img({'img': frame})['img']
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                x = transform_to_net({'img': img})['img']
                x.unsqueeze_(0)
                gpu_start = time.time()
                seg_pred, exist_pred = network(net, x)
                gpu_end = time.time()

                seg_pred = seg_pred.numpy()[0]
                exist_pred = exist_pred.numpy()

                exist = [1 if exist_pred[0, i] > 0.5 else 0 for i in range(4)]

                loop_end = time.time()

                if args.visualize:
                    visualize_start = time.time()
                    img2 = visualize(img, seg_pred, exist_pred)

                    '''
                    for i in getLane.prob2lines_CULane(seg_pred, exist, pts=30):
                        print(i)
                        #img = draw_linear_regression_lane(i, img)
                        #img = draw_polynomial_regression_lane(i, img)
                        #img = draw_lane_ransac(i, img)
                    '''
                    points = getLane.prob2lines_CULane_make(seg_pred, exist, pts=30)
                    if re_eval_sign(points) == True:
                        centerline_visualize(img, seg_pred, exist_pred)
               
                    lane_img, img2 = draw_lanes(points, img2)
                    lane_img = affine_trasform(lane_img)
                    img2= overwrap(lane_img, img2)
                    #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                    cv2.imshow('input_video', frame)
                    #cv2.imshow('transform_video', lane_img)
                    cv2.imshow("output_video", img2)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                visualize_end = time.time()
                print("visualize_runtime:", visualize_end - visualize_start, "FPS:", int(1 / (visualize_end - visualize_start)))
            else:
                break

        print("gpu_runtime:", gpu_end - gpu_start, "FPS:", int(1 / (gpu_end - gpu_start)))
        print("total_runtime:", visualize_end - loop_start, "FPS:", int(1 / (loop_end - loop_start)))
        

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
