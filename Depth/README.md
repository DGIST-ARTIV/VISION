# Get Real-time Depth Map from a Camera
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
<p align="center"><img src="https://user-images.githubusercontent.com/59161083/87302332-2b212400-c54c-11ea-8854-dafc1eab6cf6.png" width="150%" height="150%"></img></p>



## TO DO
0. ~~ROS Image topic을 받은 후, Depth 정보를 Image로 Publish~~ (20.07.11)   
1. ~~ROS를 사용하는 경우 심한 delay 발생...~~
