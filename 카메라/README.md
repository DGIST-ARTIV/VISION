# How to use FLIR Camera(Grasshopper USB3)
Author: 이  구
date: 2020.07.07

## 구성품 및 동작 확인

카메라와 박스 하나. 150만원짜리 카메라가 맞나 싶다. 덕분에 카메라가 정상적으로 작동된다는 것을 확인하는데에만 이틀이 걸렸다.   
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

이제 아래와 같이 기존의 opencv와 비슷한 방식으로 사용할 수 있다

```(python3
ret, frame = cap.read()
    img_show = cv2.resize(frame, None, fx=args.scale, fy=args.scale)
    img_show = cv2.cvtColor(img_show, cv2.COLOR_BayerGB2RGB)
```

왜인지 모르겠지만, BGR이 아니라, RGB로 바꿔주어야 이미지가 똑바로 출력된다.

### ROS2


