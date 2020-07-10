# How to use FLIR Camera(Grasshopper USB3)
Author: 이  구   
date: 2020.07.07

## 구성품 및 동작 확인

카메라와 박스 하나. 150만원짜리 카메라가 맞나 싶다. 덕분에 카메라가 정상적으로 작동된다는 것을 확인하는데에만 이틀이 걸렸다.   

카메라는 아래와 같이 생겼다. 작고 귀엽다.   

<p align="center"><img src="https://user-images.githubusercontent.com/59161083/86768884-332f1e80-c089-11ea-9b86-b5bcc539f237.jpg" width="50%" height="50%"></img></p>

뒤에는 두개의 포트가 있는데, 윗쪽의 포트는 Micro-B 타입으로 컴퓨터와 연결하면 되고, 아래의 포트는 GPIO로 보조전원용이다.   

<p align="center"><img src="https://user-images.githubusercontent.com/59161083/86769034-71c4d900-c089-11ea-8617-20ff9ad1c115.jpg" width="50%" height="50%"></img></p>

카메라 앞을 덮고 있는 덮개를 벗긴 후, 따로 구입한 c-mount 렌즈를 돌려서 끼워넣으면 된다.   

<p align="center"><img src="https://user-images.githubusercontent.com/59161083/86769198-bea8af80-c089-11ea-9039-698f4e038d30.jpg" width="50%" height="50%"></img></p>

박스의 윗면을 보면, 아래와 같은 문구가 적혀있다.   

    **Before plugging in your camera**    
    Download the Getting Started Manual and Software.   
    Go to www.flir.com/mv-getting-started   

위의 링크를 타고 들어가면 간단한 사용 설명서를 볼 수 있다. 정말 간단하다.   

카메라가 정상적으로 동작하는 것을 확인하기 위해 아래의 링크에 들어가, spinnaker sdk를 설치하자.    
https://www.flirkorea.com/support-center/iis/machine-vision/downloads/spinnaker-sdk-and-firmware-download/   

설치 후, Spin view를 실행하면 카메라가 동작하는 것을 확인할 수 있다.

## 사용법(python3, ROS2)
### python3
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


### Note
* FLIR_CAMERA(revised): 카메라가 컴퓨터와 정상적으로 연결되어 있지 않으면, FATAL log가 발생
