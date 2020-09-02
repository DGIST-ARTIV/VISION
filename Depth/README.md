# Get Real-time Depth Map from a Camera and Application to ROS
Author : 이  구 <br/>
 > reference: https://github.com/nianticlabs/monodepth2
 
## Environment Setting
OS: Ubuntu 18.04
Camera: Logitech BRIO  
pytorch: 1.2.0   
CUDA: 10.0   
cudnn: 7.6.5   
opencv: 4.3.0
GPU: RTX 2080Ti   
ROS: melodic

## Usage
하나의 카메라를 사용하여 실시간으로 depth map을 뽑을 수 있다.

#### opencv
```(python)
python3 depth_video.py --model_name [mono+stereo_640x192] --width [width] --height [height] 
```
* [mono+stereo_640x192] 값을 바꾸어 주면 다른 데이터로 학습된 모델을 사용할 수 있다. 모델의 종류 및 차이점은 [여기](https://github.com/nianticlabs/monodepth2)서 확인
* [width], [height] 값을 조절하여 depth map의 해상도를 설정할 수 있다.

#### ROS
```(python)
python ROS_monodepth.py --model_name [mono+stereo_640x192] --width [width] --height [height] 
```


## Execution Result
#### cv2를 이용하여 읽은 이미지를 이용하여 depth 추정 후, cv2로 출력 ([depth_video.py](https://github.com/DGIST-ARTIV/VISION/blob/master/Depth/depth_video.py))
<p align="center"><img src="https://user-images.githubusercontent.com/59161083/87166967-45b68b80-c307-11ea-9b86-ece82858d94d.gif" width="150%" height="150%"></img></p>

#### ROS Image topic으로 받은 원본 이미지를 이용하여 depth를 추정한 후, ROS Image topic으로 publish ([ROS_monodepth.py](https://github.com/DGIST-ARTIV/VISION/blob/master/Depth/ROS_monodepth.py))
<p align="center"><img src="https://user-images.githubusercontent.com/59161083/87181071-a7cdbb80-c31c-11ea-9169-1b0c5ea35e00.gif" width="150%" height="150%"></img></p>

#### 해상도에 따른 성능 변화
| resolution | fps |
|:--------:|:--------:|
| 720 x 480 | 약 36 fps |
| 1080 x 720 | 약 23 fps |
| 1920 x 1080 | 약 8 fps |

## Improvement
#### 기존의 ROS node graph
기존에 사용한 방식의 ros node graph는 아래와 같다. 
<p align="center"><img src="https://user-images.githubusercontent.com/59161083/87303410-18a7ea00-c54e-11ea-9a1d-4802df70a766.png" width="150%" height="150%"></img></p>

기존의 방식에서 이미지 토픽을 몇 번 Publish 하게되는지 확인해보자.     
```/usb_cam```에서 ```/detector_manager```로 */usb_cam/image_raw* topic을 **한번**, ```/detector_manager```에서 ```/DetectedImg```로 *yolov3/image_raw* topic을 **한번**, 마지막으로 ```/DetectedImg```에서 */YOLO_RESULT* topic을 **한번** publish 하게 된다. 

usb_cam을 통해 받은 이미지를, monodepth2로 depth map을 얻기 위해,```/usb_cam/image_raw```에서 ```/DepthMap```으로 Image topic을 한번 더 받아오게 된다. 아직 구현하지는 못했지만, 이미지의 최종 정보에 DepthMap을 사용하여 추정한 거리 정보를 포함시킬 것이기 때문에, 최소 한번 이상의 추가적인 publish가 필요할 것이다.

이 점을 고려하면, **이미지 한 장**이 노드 사이를 **5번 이상** 이동하게 되는 것이다. 이때, 동시에 처리하는 데이터의 양이 늘어날수록(bandwidth가 높아질수록), ROS 자체의 delay가 심해진다.  
이로인해, 전체적인 시스템을 수정하게 되었다.

#### 수정된 ROS node graph

<p align="center"><img src="https://user-images.githubusercontent.com/59161083/87302562-8eab5180-c54c-11ea-9f3b-ee6c451d616e.png" width="150%" height="150%"></img></p>

새롭게 수정된 방식에서는, ```/usb_cam```에서 ```/detector_manager```로 */usb_cam/image_raw* topic을 **한번**, ```/detector_manager```에서 ```/PostProcessing```으로 */yolov3/image_raw* topic을 **한번** publish 한 후, 한 노드 안에서 detect 결과의 시각화와 depth map 추정이 동시에 이루어지게 된다.    

## TO DO
0. ~~ROS Image topic을 받은 후, Depth 정보를 Image로 Publish~~ (20.07.11)   
0. ~~ROS를 사용하는 경우 심한 delay 발생...~~ (20.07.13)
0. ~~DepthMap과 Bbox를 이용하여 인식된 객체의 거리 추정하기~~ (20.07.14)
0. ~~DepthMap estimation 결과를 출력하는 과정에서 normalization 없애기~~ (20.07.14)
0. 인식된 객체의 실제 위치와 DepthMap의 Bbox pixel값 평균 비교하기
