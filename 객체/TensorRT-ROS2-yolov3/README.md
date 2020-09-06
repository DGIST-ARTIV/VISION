# How to use TensorRT-yolov3 with ROS2
Author: 이  구
date: 2020.07.21

### TensorRT yolov3
> reference: https://github.com/jkjung-avt/tensorrt_demos   
yolov3.weights -> yolov3.onnx -> yolov3.trt 로 변환해야 한다. 그 과정은 위의 링크를 참고.   

### TensorRT yolov3 with ROS2
TensorRT-ROS2-yolov3 안의 파일들을 다운 받은 후, catkin_ws/src 로 옮긴다. catkin_make 후 사용하면 된다.   

#### trt_yolov3_node
ROS의 Image 타입 message를 받은 후, trt-yolov3로 object들을 detect한 후, 그 이미지를 Image 타입의 메세지(/TRT_yolov3/image_result)로, 인식된 물체들을 BoundingBoxes 타입의 메세지(/TRT_yolov3/Bbox)로 publish한다. 이때, BoundingBoxes는 custom message로, BoundingBox의 array이다. BoundingBox 역시 custom message로, 아래와 같이 구성되어 있다.   

    string Class
    float64 probability
    int64 xmin
    int64 ymin
    int64 xmax
    int64 ymax

**custom message는 ros_bridge를 통해 전달되지 않는다.**


### To Do
0. ~~약 60 fps였던 trt-yolov3가 ROS2에 적용하면 20 fps로 실행됨~~ (2020.07.22)   
성능이 말도안되게 떨어지는 것이 이해할 수 없어, 각 과정에서 소요되는 시간을 측정해보았다.     
callback 함수의 기능을 분리해서 살펴보면, 아래와 같다.   
    1. ROS2 Image type의 message를 cv2 image 형식으로 변환 (cv_bridge 사용)
    2. cv2 형식의 이미지를 통해 trt-yolov3 detect 실행 -> bbox 얻음.
    3. BoundingBoxes 타입의 메세지로 해당 이미지에 존재하는 객체들의 Bbox 정보를 mapping 및 publish.
    4. Bbox 정보들을 이미지에 시각화한 후, 해당 이미지를 ROS2 Image type으로 변환하여 publish
    
    적폐는 4번. cv2 형식의 이미지를 ROS Image 타입으로 바꾸는 과정에서 약 0.03초가 소요된다. 이미지 대신 bbox만 publish 하기로 결정.    

0. ~~launch 파일 만들기~~   
0. ~~custom object 학습 및 확인~~ (2020.07.24)   
<p align="center"><img src="https://user-images.githubusercontent.com/59161083/88577129-73c6fa00-d081-11ea-8b19-3cc44c97e6e4.jpg" width="50%" height="50%"></img></p>
        위의 사진은 사람만 학습시킨 후, 테스트 한 결과이다. 약 130 fps.

