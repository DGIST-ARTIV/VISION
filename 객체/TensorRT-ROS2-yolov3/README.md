# How to use TensorRT-yolov3 with ROS2
Author: 이  구
date: 2020.07.22

### TensorRT yolov3
> reference: https://github.com/jkjung-avt/tensorrt_demos
yolov3.weights -> yolov3.onnx -> yolov3.trt 로 변환해야 한다. 그 과정은 위의 링크를 참고.


### TensorRT yolov3 with ROS2
TensorRT-ROS2-yolov3 안의 파일들을 다운 받은 후, catkin_ws/src 로 옮긴다. catkin_make 후 사용하면 된다.

#### trt_yolov3_node
ROS의 Image 타입 message를 받은 후, trt-yolov3로 object들을 detect한 후, 그 이미지를 Image 타입의 메세지로, 인식된 물체들을 BoundingBoxes 메세지 타입으로 publish한다. 이때, BoundingBoxes는 custom message로, BoundingBox의 array이다. BoundingBox 역시 custom message로, 아래와 같이 구성되어 있다.   

    string Class
    float64 probability
    int64 xmin
    int64 ymin
    int64 xmax
    int64 ymax

**custom message는 ros_bridge를 통해 전달되지 않는다.**

