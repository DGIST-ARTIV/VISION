# How to use FLIR Camera(Grasshopper USB3)
Author: 이  구
date: 2020.07.07

## 사용법
build는 무시하고, src의 파일들을 다운 받은 후, 자신의 catkin_ws/src 로 옮긴다. catkin_make 후 사용하면 된다.


## 간단한 패키지 설명(추후 update)
### usb_cam
말 그대로, usb_cam을 사용하기 위해 사용하는 패키지이다. launch 파일을 실행하게 되면, 컴퓨터에 연결된 usb_cam을 통해 이미지를 읽은 후, ROS의 Image 타입 message로 publish하게 된다. 

### movie_publisher
usb_cam이 아닌, 동영상 파일을 이용하기 위해 찾은 패키지이다. launch 파일을 실행하면 원하는 동영상 파일을 한 프레임씩 ROS의 Image 타입 message로 publish하게 된다.

### video_stream_opencv
위의 패키지와 마찬가지로, usb_cam이 아닌 동영상 파일을 이용하기 위해 찾은 패키지이다. 60 fps로 영상을 publish 한 후 사용해보고 싶어 cpp로 짜여진 다른 패키지를 찾아서 사용해보았지만, fps 설정을 바꿔주어도 영상을 30fps로 뱉는다. 

### yolov3_pytorch_ros
ROS의 Image 타입 message를 받은 후, yolo v3로 object들을 detect한 후, 그 이미지와 인식된 물체들을 bboxes 메세지 타입으로 publish한다. 이때, bboxes는 custom message

### YOLO

python으로 FLIR 카메라에서 이미지를 받아온 후, 화면에 출력해보자.   
FLIR 카메라는 기존에 쓰던 웹캠과는 달리, usb를 연결하자마자 카메라로 인식되지는 않는다.   
즉, cv2.videocapture() 같은 함수를 사용해도 이미지를 받아올 수 없다.   
FLIR 카메라를 사용하기 위해, PySpin, EasyPySpin 라이브러리를 설치하자.

> PySpin: FLIR official python library   
> EasyPySpin: unofficial wrapper for FLIR Spinnaker SDK. This wrapper provides much the same way as the OpenCV VideoCapture class.   

이제 아래와 같이 FLIR 카메라를 사용할 수 있다.   

```(python3)
cap = EasyPySpin.VideoCapture(0)
cap.cam.PixelFormat.SetValue(PySpin.PixelFormat_BayerGB8)
```

이때, pixel format을 정해주어야하는데 우리는 SpinView에서 확인한 pixel format으로 설정해주었다.   

```(python3)
cap.cam.PixelFormat.SetValue(PySpin.PixelFormat_BayerGB8)
```

이제 아래와 같이 기존의 opencv와 비슷한 방식으로 사용할 수 있다.   

```(python3
ret, frame = cap.read()
img_show = cv2.resize(frame, None, fx=args.scale, fy=args.scale)
img_show = cv2.cvtColor(img_show, cv2.COLOR_BayerGB2RGB)
```

왜인지 모르겠지만, BGR이 아니라, RGB로 바꿔주어야 이미지가 똑바로 출력된다.   

전체 코드는 [여기](https://github.com/DGIST-ARTIV/VISION/blob/master/%EC%B9%B4%EB%A9%94%EB%9D%BC/get_image_from_flir_camera.py)   

### ROS2
cv2 format의 이미지를 ROS2의 Image형식으로 바꿔주기 위해 CvBridge를 사용해야 한다. 이를 사용하기 위해, cv_bridge 라이브러리를 설치하자.   
ros2를 사용해야 하니 위의 코드에서 rclpy를 import하자.   

이후, 계속해서 실행될 img_callback 함수를 만들어 주었다.   

```(python3)
ret, img = self.cap.read()
img = cv2.cvtColor(img, cv2.COLOR_BayerGB2RGB)
img = cv2.resize(img, dsize = (args.width, args.height))
temp=CvBridge().cv2_to_imgmsg(img, encoding = 'bgr8')
self.publisher_.publish(temp)
```

위의 과정을 통해 cv2 형식의 이미지를 ROS2의 image message type의 형식으로 만들어준 후, publish 하게 된다.   

전체 코드는 [여기](https://github.com/DGIST-ARTIV/VISION/blob/master/%EC%B9%B4%EB%A9%94%EB%9D%BC/FLIR_CAMERA(revised).py) 

위의 코드를 그대로 실행하게 되면, FLIR_ImgPublisher라는 노드에서 FLIR_IMAGE 라는 topic을 publish하게 된다.

<p align="center"><img src="https://user-images.githubusercontent.com/59161083/86770545-c8cbad80-c08b-11ea-86b6-f01d63071880.png" width="100%" height="100%"></img></p>

