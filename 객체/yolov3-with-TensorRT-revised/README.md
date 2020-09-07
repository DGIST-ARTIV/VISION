# How to use yolov3 with ROS
Author: 이  구   
date: 2020.09.07   

## 사용법
build는 무시하고, src의 파일들을 다운 받은 후, 자신의 catkin_ws/src 로 옮긴다. catkin_make 후 사용하면 된다.

## 간단한 패키지 설명(추후 update 예정)
### usb_cam
말 그대로, usb_cam을 사용하기 위해 사용하는 패키지이다. launch 파일을 실행하게 되면, 컴퓨터에 연결된 usb_cam을 통해 이미지를 읽은 후, ROS의 Image 타입 message로 publish하게 된다.   

### movie_publisher
usb_cam이 아닌, 동영상 파일을 이용하기 위해 찾은 패키지이다. launch 파일을 실행하면 원하는 동영상 파일을 한 프레임씩 ROS의 Image 타입 message로 publish하게 된다.   

### video_stream_opencv
위의 패키지와 마찬가지로, usb_cam이 아닌 동영상 파일을 이용하기 위해 찾은 패키지이다. 60 fps로 영상을 publish 한 후 사용해보고 싶어 cpp로 짜여진 다른 패키지를 찾아서 사용해보았지만, fps 설정을 바꿔주어도 영상을 30fps로 뱉는다.    

### yolov3_pytorch_ros
ROS의 Image 타입 message를 받은 후, yolo v3로 object들을 detect한 후, 그 이미지를 Image 타입의 메세지로, 인식된 물체들을 BoundingBoxes 메세지 타입으로 publish한다. 이때, BoundingBoxes는 custom message로, BoundingBox의 array이다. BoundingBox 역시 custom message로, 아래와 같이 구성되어 있다.   

    string Class
    float64 probability
    int64 xmin
    int64 ymin
    int64 xmax
    int64 ymax


### YOLO
위의 yolov3_pytorch_ros 패키지에서 publish한 정보를 활용하는 패키지이다. YOLO 패키지는 두 개의 노드로 구성되어 있는데, 하나는 BboxSubscriber, 다른 하나는 ImgPreprocessing 이다.

BboxSubscriber 노드는 yolov3_pytorch_ros 패키지에서 publish한 BoundingBoxes 메세지 정보를 가공한 후, 터미널에 출력해주는 역할을 한다. 인식된 object의 class, center point, width, height 값을 출력하게 된다.   

ImgPreprocessing 노드는 yolov3_pytorch_ros 패키지에서 publish한 Image 메세지와 BoundingBoxes 메세지를 이용하여 시각화하는 역할을 한다. opencv를 이용하여 물체가 인식된 영역에 해당 class에 해당하는 색으로 사각형을 그린 후, 해당 객체의 class와 현재 위치(left, center, right)를 텍스트로 표시하게 된다.

### node graph
#### version1
<p align="center"><img src="https://user-images.githubusercontent.com/59161083/86787084-b0658e00-c09f-11ea-9872-0b4174d05446.png" width="100%" height="100%"></img></p>

위의 DetectedImg는 Bbox와 image_raw topic을 동시에 받은 후, callback을 실행한다. 이를 구현하기 위해 ApproximateTimeSynchronizer를 사용하였다. 자세한 내용은 [여기](https://github.com/DGIST-ARTIV/VISION/blob/master/%EA%B0%9D%EC%B2%B4/yolov3_pytorch+ROS/src/YOLO/src/ImgPostprocessing.py)를 참고.


#### version2
<p align="center"><img src="https://user-images.githubusercontent.com/59161083/87340756-7b67a880-c583-11ea-8f1c-fcfce286357b.png" width="100%" height="100%"></img></p>

이후, 위와 같은 구조로 변경하였다. monodepth2를 이용하는 depthmap estimation이 추가되었다. 자세한 과정은 [여기](https://github.com/DGIST-ARTIV/VISION/blob/master/Depth/README.md)를 참고.




