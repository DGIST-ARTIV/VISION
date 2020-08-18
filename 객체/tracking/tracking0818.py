# Notes!
# Last modification 0815

# Update List
# Date: 0814
# Author: 이 구
# get_current_obj() 함수 구조 수정
# 애매한 상황이 있으면 id 초기화 (모든 obj의 id를 None으로 설정)
# 프레임이 충분히 많고, dist_th를 잘 설정하면 거의 발생하지 않을듯

# Date: 0815
# Author: 이 구
# 이전 프레임을 이용해 현재 프레임의 객체 정보를 보정해주는 함수를 구현하기 전에, id 관리 먼저 똑바로 해야 함.
# 적어도 열 프레임 안에서 동일한 id가 부여하지 않도록 해야 함.
# Circular Queue 형태로 관리. 열 프레임 안에 새로운 물체가 100번 이상 나오는게 아니면, id가 겹칠 일 X
# id_manager를 만들 때, size를 설정할 수 있도록 함 
# obj class의 list가 정렬될 수 있도록 수정.

# Date: 0816
# Author: 이 구
# 이전 프레임을 이용해 현재 프레임을 보상하는 기능 구현
# Tracking 하고 싶은 class 별로 Queue_for_tracking을 만들어서 사용하면 됌.
# 이때, 동일한 id manager를 사용하면 모든 객체를 동일한 id manager로 관리하고, 다른 id manager를 사용하면 class별로 id를 관리한다.

# Date: 0817
# Author: 이 구
# argparser를 이용해서 category_num, dist_th, compensate_th 조정할 수 있도록 수정.
# KCITY custom 완료, 총 14가지 물체를 종류별로 tracking 함.

# To Do List
# ROS Publish 구현

import os
import rclpy
from std_msgs.msg import String, Int16MultiArray, MultiArrayDimension
import math
import gc
import time
import argparse


KCITY_CUSTOM_LIST = ['car', 'crossWalk_sign', 'bicycle_sign', 'bust_sign', 'construction_sign', 'parking_sign',
                     'kidSafeSero_sign', 'busArrowDown_sign', 'trafficLightRedYellow', 'trafficLightGreenLeft',
                     'trafficLightYellow', 'trafficLightGreen', 'trafficLightRed', 'trafficLightRedLeft']

class id_manager:
    def __init__(self, size=100):
        self._size = size
        self.header_ptr = 0
        self.id_list = [x for x in range(self._size)]
        self.used_id_list = []
        self._score = []
        for i in range(self._size):
            self._score.append(0)

    def give_id(self):
        num = self.id_list[self.header_ptr]
        self.increase_ptr()
        while num in self.used_id_list:
            num = self.id_list[self.header_ptr]
            self.increase_ptr()
        self.used_id_list.append(num)
        return num

    def receive_id(self, num):
        if num != None:
            self.used_id_list.remove(num)
        
    def increase_ptr(self):
        self.header_ptr += 1
        if self.header_ptr == self._size:
            self.header_ptr = 0
        
    def decrease_score(self, id):
        self._score[id] -= 1
        if self._score[id] < 0:
            self._score[id] = 0

    def clean_used_id_list(self):
        for i in self.used_id_list:
            if self._score[i] == 0:
                self.receive_id(i)

class Queue_for_tracking:
    def __init__(self, id_manager, size=10):
        self._size = size
        self._queue = [None] * self._size
        self._start_flag = False
        self._idmanager = id_manager

    def add_queue(self, item):
        if self._start_flag == False:
            for i in range(self._size):
                for i in range(self._size - 1):
                    if i < self._size:
                        self._queue[self._size - 1 - i] = self._queue[self._size - 2 - i]
                self._queue[0] = item
            self._start_flag = True
        else:
            for i in range(self._size - 1):
                if i < self._size:
                    self._queue[self._size - 1 - i] = self._queue[self._size - 2 - i]
            self._queue[0] = item

    def display(self):
        print(sorted(self._queue[0]))

    def get_current_objs(self):
        past_bboxes = self.make_predicted_bboxes()
        current_bboxes = self._queue[0]
        if past_bboxes != None and current_bboxes != None:
            if len(past_bboxes) == 0 and len(current_bboxes) != 0:
                for i in range(len(current_bboxes)):
                    current_bboxes[i].id = self._idmanager.give_id()
                return True

            elif len(past_bboxes) != 0 and len(current_bboxes) == 0:
                return True

            elif len(past_bboxes) == 0 and len(current_bboxes) == 0:
                return True
            
            else:
                check = 0
                # Check for initialize
                for i in range(len(past_bboxes)):
                    if past_bboxes[i].id == None:
                        check += 1
                    else:
                        check += 0
                if check == len(past_bboxes):
                    for i in range(len(current_bboxes)):
                        current_bboxes[i].id = self._idmanager.give_id()
                    return True
                
                # generate dist_table
                dist_table = []
                for i in range(len(past_bboxes)):
                    temp = []
                    for j in range(len(current_bboxes)):
                        temp.append(current_bboxes[j].cal_dist(past_bboxes[i]))
                    dist_table.append(temp)
               
                # generate result
                result = []
                for j in range(len(current_bboxes)):                
                    temp = []
                    for i in range(len(past_bboxes)):
                        temp.append(dist_table[i][j])
                    
                    if min(temp) < dist_th:
                        result.append(temp.index(min(temp)))
                    
                    else:
                        result.append(j + 1000)

                # check if or not result has duplicated value 
                if has_duplicates(result):
                    # initialize all obj's id
                    for i in range(len(current_bboxes)):
                        current_bboxes[i].id = None
                    return True

                # if new object detected in current frame, give new id
                for j in range(len(current_bboxes)):
                    if result[j] < 1000:
                        current_bboxes[j].id = past_bboxes[result[j]].id
                    else:
                        current_bboxes[j].id = self._idmanager.give_id()
                
                past_id_lst = []
                current_id_lst = []

                # compensate with past 10 frame's information
                for i in range(len(current_bboxes)):
                    current_id_lst.append(current_bboxes[i].id)

                for j in range(len(past_bboxes)):
                    past_id_lst.append(past_bboxes[j].id)

                for i in range(len(past_bboxes)):
                    if past_id_lst[i] not in current_id_lst:
                        temp = self.find_recent_obj(past_id_lst[i])
                        temp.isReal = False
                        self._idmanager._score[temp.id] -= 1
                        current_bboxes.append(temp)
                return True
        else:
            return
    
    # Function for compensation (update score, find_recent_obj, make_predicted_bboxes)
    def update_score(self):
        self._idmanager._score = []
        for i in range(self._idmanager._size):
            self._idmanager._score.append(0)

        for i in range(1, len(self._queue)):
            for j in range(len(self._queue[-i])):
                if self._queue[-i][j].id != None and self._queue[-i][j].isReal == True:
                    if self._idmanager._score[self._queue[-i][j].id] < compensate_th:
                        self._idmanager._score[self._queue[-i][j].id] += 1

    def find_recent_obj(self, id):
        num = -121456
        for i in range(1, len(self._queue)):
            for j in range(len(self._queue[i])):
                num = self._queue[i][j].id
                if num == id:
                    return self._queue[i][j]
        return None

    def make_predicted_bboxes(self):
        self.update_score()
        result = []
        for i in range(len(self._idmanager._score)):
            if self._idmanager._score[i] > 0:
                temp = self.find_recent_obj(i)
                if temp == None:
                    continue
                else:
                    result.append(temp)
        if len(result) == None:
            return None
        return result

    def execute(self, list):
        self.add_queue(list)
        self.get_current_objs()
        self.display()
        self._idmanager.clean_used_id_list()

class obj:
    def __init__(self, xmin, xmax, ymin, ymax, clss, conf):
        self.center_x = (xmin + xmax) // 2
        self.center_y = (ymin + ymax) // 2
        self.width = xmax - xmin
        self.height = ymax - ymin
        self.clss = get_cls_dict(category_num)[clss]
        self.conf = conf
        self.id = None
        self.isReal = True

    def __repr__(self):
        return "type: " + str(self.clss) + " id: " + str(self.id) + " is real: " + str(self.isReal)+" center pt: (" + str(self.center_x) + "," + \
               str(self.center_y) + "), width: " + str(self.width) + ", height: " + str(
            self.height) + "), conf: " + str(self.conf) + "\n"

    def __lt__(self, other):
        if self.id == None or other.id == None:
            return True
        return self.id < other.id

    def info(self):
        print("This is one object of 'obj class'. It includes the type of object, id of object, center point of object, size of object, confidence of object.")

    def cal_dist(self, other):
        return math.sqrt((self.center_x - other.center_x) ** 2 + (self.center_y - other.center_y) ** 2)

def get_cls_dict(category_num):
    """Get the class ID to name translation dictionary."""
    if category_num == 80:
        return {i: n for i, n in enumerate(COCO_CLASSES_LIST)}

    elif category_num == 14:
        return {i: n for i, n in enumerate(KCITY_CUSTOM_LIST)}

def has_duplicates(seq):
    return len(seq) != len(set(seq))

def Bbox_callback(msg: Int16MultiArray):
    now = time.time()
    data = msg.data
    list_0 = []
    list_1 = []
    list_2 = []
    list_3 = []
    list_4 = []
    list_5 = []
    list_6 = []
    list_7 = []
    list_8 = []
    list_9 = []
    list_10 = []
    list_11 = []
    list_12 = []
    list_13 = []
    
    if data is not None:
        num_of_obj = len(data) // 6
        for i in range(num_of_obj):
            clss = data[6 * i]
            if clss == 7:
                clss = 2
            conf = data[6 * i + 1]
            xmin = data[6 * i + 2]
            ymin = data[6 * i + 3]
            xmax = data[6 * i + 4]
            ymax = data[6 * i + 5]
            temp = obj(xmin, xmax, ymin, ymax, clss, conf)
            if temp.clss == 'car':
                list_0.append(temp)
            if temp.clss == 'crossWalk_sign':
                list_1.append(temp)
            if temp.clss == 'bicycle_sign':
                list_2.append(temp)
            if temp.clss == 'bust_sign':
                list_3.append(temp)
            if temp.clss == 'construction_sign':
                list_4.append(temp)
            if temp.clss == 'parking_sign':
                list_5.append(temp)
            if temp.clss == 'kidSafeSero_sign':
                list_6.append(temp)
            if temp.clss == 'busArrowDown_sign':
                list_7.append(temp)
            if temp.clss == 'trafficLightRedYellow':
                list_8.append(temp)
            if temp.clss == 'trafficLightGreenLeft':
                list_9.append(temp)
            if temp.clss == 'trafficLightYellow':
                list_10.append(temp)
            if temp.clss == 'trafficLightGreen':
                list_11.append(temp)
            if temp.clss == 'trafficLightRed':
                list_12.append(temp)
            if temp.clss == 'trafficLightRedLeft':
                list_13.append(temp)
    
    Queue_0.execute(sorted(list_0))
    Queue_1.execute(sorted(list_1))
    Queue_2.execute(sorted(list_2))
    Queue_3.execute(sorted(list_3))
    Queue_4.execute(sorted(list_4))
    Queue_5.execute(sorted(list_5))
    Queue_6.execute(sorted(list_6))
    Queue_7.execute(sorted(list_7))
    Queue_8.execute(sorted(list_8))
    Queue_9.execute(sorted(list_9))
    Queue_10.execute(sorted(list_10))
    Queue_11.execute(sorted(list_11))
    Queue_12.execute(sorted(list_12))
    Queue_13.execute(sorted(list_13))

    gc.collect()
    print(time.time()-now)
    print("----------------------------------------------------------------------------")

def parse_args():
    desc = ('Tracking the obj of specific class')
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dist_th', type=int, default='150',
                        help='select the distance threshold value')
    parser.add_argument('--compensate_th', type=int, default='3',
                    help='select the compensation threshold value')
    args = parser.parse_args()
    return args

node = 0
IDM = id_manager(200)

Queue_0 = Queue_for_tracking(IDM,11)
Queue_1 = Queue_for_tracking(IDM,11)
Queue_2 = Queue_for_tracking(IDM,11)
Queue_3 = Queue_for_tracking(IDM,11)
Queue_4 = Queue_for_tracking(IDM,11)
Queue_5 = Queue_for_tracking(IDM,11)
Queue_6 = Queue_for_tracking(IDM,11)
Queue_7 = Queue_for_tracking(IDM,11)
Queue_8 = Queue_for_tracking(IDM,11)
Queue_9 = Queue_for_tracking(IDM,11)
Queue_10 = Queue_for_tracking(IDM,11)
Queue_11 = Queue_for_tracking(IDM,11)
Queue_12 = Queue_for_tracking(IDM,11)
Queue_13 = Queue_for_tracking(IDM,11)


def main():
    global args
    global category_num
    global dist_th
    global compensate_th
    rclpy.init(args=None)
    
    args = parse_args()
    category_num = 14
    dist_th = args.dist_th
    compensate_th = args.compensate_th

    node = rclpy.create_node('get_bbox')
    sub = node.create_subscription(Int16MultiArray, '/TRT_yolov3/Bbox', Bbox_callback)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
