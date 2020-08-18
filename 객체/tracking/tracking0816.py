# Notes!
# Last modification 0815

# Update List
# Date: 0814
# get_current_obj() 함수 구조 수정
# 애매한 상황이 있으면 id 초기화 (모든 obj의 id를 None으로 설정)
# 프레임이 충분히 많고, dist_th를 잘 설정하면 거의 발생하지 않을듯

# Date: 0815
# 이전 프레임을 이용해 현재 프레임의 객체 정보를 보정해주는 함수를 구현하기 전에, id 관리 먼저 똑바로 해야 함.
# 적어도 열 프레임 안에서 동일한 id가 부여하지 않도록 해야 함.
# Circular Queue 형태로 관리. 열 프레임 안에 새로운 물체가 100번 이상 나오는게 아니면, id가 겹칠 일 X
# id_manager를 만들 때, size를 설정할 수 있도록 함
# obj class의 list가 정렬될 수 있도록 수정.

# To Do List
# 이전 열 프레임의 정보를 이용해 현재 프레임을 보상해주는 함수(이름: compensate) 구현
# Queue_for_tracking 클래스 내부에 구현 예정
# get_current_objs 함수를 실행하기 전, compensate 실행
# get_current_objs를 실행하면 현재 frame이 수정된다. 이때, obj들이 실제 detect 된 것인지, 보상받은 것인지 구분하기 위해 flag(변수명: isreal) 사용 예정
# 이전 열 프레임에서 인식된 모든 물체 중, isreal= True 인 물체의 id가 나오는 횟수를 count.
# count가 6? 이상일 경우 isreal=False 인 obj를 현재 프레임에 추가.

# Problem
# NMS가 똑바로 안되는 경우가 많음... -> trt_yolo에서 nms_th 조정해보기

import os
import rclpy
from std_msgs.msg import String, Int16MultiArray, MultiArrayDimension
import math
import gc
import time

category_num = 80
dist_th = 150
compensate_th = 3
frame_num = 0

COCO_CLASSES_LIST = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
                     'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                     'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
                     'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                     'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                     'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                     'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable',
                     'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                     'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                     'toothbrush']

KCITY_CUSTOM_LIST = ['car', 'crossWalk_sign', 'bicycle_sign', 'bust_sign', 'construction_sign', 'parking_sign',
                     'kidSafeSero_sign', 'busArrowDown_sign', 'trafficLightRedYellow', 'trafficLightGreenLeft',
                     'trafficLightYellow', 'trafficGreen', 'trafficLightRed', 'trafficLightRedLeft']

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
            self._score[num] = 0
        return True

    def increase_ptr(self):
        self.header_ptr += 1
        if self.header_ptr == self._size:
            self.header_ptr = 0
        return True

class Queue_for_tracking:
    def __init__(self, size=10):
        self._size = size
        self._queue = [None] * self._size

    def add_queue(self, item):
        for i in range(self._size - 1):
            if i < self._size:
                self._queue[self._size - 1 - i] = self._queue[self._size - 2 - i]
        self._queue[0] = item

    def display(self):
        print(self._queue[0])

    def get_current_objs(self):
        past_bboxes = self._queue[1]
        current_bboxes = self._queue[0]
        predicted_bboxes = self.predict_bboxes()

        if past_bboxes != None and current_bboxes != None:
            if len(past_bboxes) == 0 and len(current_bboxes) != 0: # case 1
                print("Case1")
                if len(predicted_bboxes) == 0:
                    print("Case1-1")
                    for i in range(len(current_bboxes)):
                        current_bboxes[i].id = IDM.give_id()
                    return True
                else:
                    ## current_bboxes 랑 predict_bboxes 비교해서 id 부여 후 return
                    print("Case1-2")
                    current_bboxes = predict_bboxes

                    pass

            elif len(past_bboxes) != 0 and len(current_bboxes) == 0: # case 2
                print("Case2")
                if len(predicted_bboxes) == 0:
                    print("Case2-1")
                    for i in range(len(past_bboxes)):
                        IDM.receive_id(past_bboxes[i].id)
                    return True
                else:
                    print("Case2-2")
                    ## current_bboxes 랑 predict_bboxes 비교해서 id 부여 후 return
                    pass

            elif len(past_bboxes) == 0 and len(current_bboxes) == 0: # case 3
                print("Case3")
                if len(predicted_bboxes) == 0:
                    print("Case3-1")
                    return True
                else:
                    print("Case3-2")
                    ## current_bboxes 랑 predict_bboxes 비교해서 id 부여 후 return
                    pass

            else: # case 4
                print("Case4")
                if len(predicted_bboxes) == 0:
                    print("Case4-1")
                    check = 0
                    for i in range(len(past_bboxes)):
                        if past_bboxes[i].id == None:
                            check += 1
                        else:
                            check += 0
                    if check == len(past_bboxes):
                        print("after initializing")
                        for i in range(len(past_bboxes)):
                            IDM.receive_id(past_bboxes[i].id)
                        for i in range(len(current_bboxes)):
                            current_bboxes[i].id = IDM.give_id()
                        return True
                    # 모든 객체에 새 id 부여

                    # generate dist_table
                    dist_table = []
                    for i in range(len(past_bboxes)):
                        temp = []
                        for j in range(len(current_bboxes)):
                            temp.append(current_bboxes[j].cal_dist(past_bboxes[i]))
                        dist_table.append(temp)
                    #print("dist_table", dist_table)

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
                        print("initialize all obj's id")
                        for j in range(len(past_bboxes)):
                            IDM.receive_id(past_bboxes[j].id)

                        for i in range(len(current_bboxes)):
                            current_bboxes[i].id = None

                        return True

                    # if new object detected in current frame, give new id
                    for j in range(len(current_bboxes)):
                        if result[j] < 1000:
                            current_bboxes[j].id = past_bboxes[result[j]].id
                        else:
                            current_bboxes[j].id = IDM.give_id()

                    # if existing object disappears, receive id
                    past_id_lst = []
                    current_id_lst = []

                    for i in range(len(current_bboxes)):
                        current_id_lst.append(current_bboxes[i].id)

                    for j in range(len(past_bboxes)):
                        past_id_lst.append(past_bboxes[j].id)

                    for i in range(len(past_bboxes)):
                        if past_id_lst[i] not in current_id_lst:
                            IDM.receive_id(past_id_lst[i])

                    return True
                else:
                    print("Case4-2")
                    ## current_bboxes 랑 predict_bboxes 비교해서 id 부여 후 return
                    pass
        else:
            return

    def update_score(self):
        for i in range(1, len(self._queue)):
            #print(self._queue)
            for j in range(len(self._queue[-i])):
                if self._queue[-i][j].id != None and self._queue[-i][j].isReal == True:
                    if IDM._score[self._queue[-i][j].id] < compensate_th:
                        IDM._score[self._queue[-i][j].id] += 1
        return True

    def select_for_compensate(self):
        self.update_score()
        result = []
        print("scores: ", IDM._score)
        for i in range(len(IDM._score)):
            if IDM._score[i] > 0:
                result.append(i)
        return result

    def find_recent_obj(self, id):
        num = -121456
        for i in range(1, len(self._queue)):
            for j in range(len(self._queue[i])):
                num = self._queue[i][j].id
                if num == id:
                    return self._queue[i][j]
        return None

    def compensate(self):
        current_id_lst = []
        result = []
        for i in range(len(self._queue[0])):
            current_id_lst.append(self._queue[0][i].id)
        comp_list = self.select_for_compensate()
        for i in comp_list:
            if i not in current_id_lst:
                obj = self.find_recent_obj(i)
                if obj == None:
                    continue
                obj.isReal = False
                result.append(obj)
        return result

    def predict_bboxes(self):
        result = []
        comp_list = self.select_for_compensate()
        for i in comp_list:
            if i not in current_id_lst:
                obj = self.find_recent_obj(i)
                if obj == None:
                    continue
                obj.isReal = False
                result.append(obj)
        return result

    def execute(self, list):
        self.add_queue(sorted(list))
        self.get_current_objs()
        self.update_score()
        self.compensate()
        self.clean_score()
        self.display()
        return True

    def start(self, list):
        self.add_queue(sorted(list))
        self.display()
        return True

class obj:
    def __init__(self, xmin, xmax, ymin, ymax, clss, conf):
        self.center_x = (xmin + xmax) // 2
        self.center_y = (ymin + ymax) // 2
        self.width = xmax - xmin
        self.height = ymax - ymin
        self.clss = get_cls_dict(category_num)[clss]
        self.conf = conf
        self.id = None

        #for test
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

    def simple_comp(self, other):
        if self.cal_dist(other) < 100:
            return True
        return False

    def TruetoFalse(self):
        if self.isReal == False:
            print("fuck you idiot")
        self.isReal = False
        return self

def get_cls_dict(category_num):
    """Get the class ID to name translation dictionary."""
    if category_num == 80:
        return {i: n for i, n in enumerate(COCO_CLASSES_LIST)}

    elif category_num == 14:
        return {i: n for i, n in enumerate(KCITY_CUSTOM_LIST)}


def has_duplicates(seq):
    return len(seq) != len(set(seq))

frame_num = 0
def Bbox_callback(msg: Int16MultiArray):
    global frame_num
    frame_num += 1
    data = msg.data
    car_list = []
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
                car_list.append(temp)

    if frame_num < 11:
        print("start")
        CarQueue.start(sorted(car_list))
    else:
        print("not start")
        CarQueue.execute(sorted(car_list))

    #print("# of obj: ", len(IDM.used_id_list))
    gc.collect()
    print("----------------------------------------------------------------------------")

node = 0

CarQueue = Queue_for_tracking(11)
IDM = id_manager()

def main():
    rclpy.init(args=None)
    node = rclpy.create_node('get_bbox')
    sub = node.create_subscription(Int16MultiArray, '/TRT_yolov3/Bbox', Bbox_callback)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
