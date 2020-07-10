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
웹캠을 사용하여 실시간으로 depth map을 뽑을 수 있다.

```(python)
python3 depth_video.py --model_name mono+stereo_640x192 --width 1080 --height 720 
```
* --model_name 뒤의 값을 바꾸어 주면 다른 모델을 사용할 수 있다. 모델의 종류 및 차이점은 [여기](https://github.com/nianticlabs/monodepth2)서 확인
* --width, --height 뒤의 값을 바꿔주면 해당 사이즈로 resize 된다.



## Execution Result
### cv2를 이용하여 읽은 이미지를 cv2로 출력한 결과(depth_video.py)
<p align="center"><img src="https://user-images.githubusercontent.com/59161083/87166967-45b68b80-c307-11ea-9b86-ece82858d94d.gif" width="150%" height="150%"></img></p>

### ROS Image topic으로 받은 원본 이미지를 이용하여 depth를 추정한 후, ROS Image topic으로 publish한 결과(ROS_monodepth.py)
<p align="center"><img src="https://user-images.githubusercontent.com/59161083/87181071-a7cdbb80-c31c-11ea-9169-1b0c5ea35e00.gif" width="150%" height="150%"></img></p>

### 해상도에 따른 성능 변화
| resolution | fps |
|:--------:|:--------:|
| 720 x 480 | 약 36 fps |
| 1080 x 720 | 약 23 fps |
| 1920 x 1080 | 약 8 fps |

## TO DO
~~0. ROS Image topic을 받은 후, Depth 정보를 Image로 Publish~~ (20.07.11)   
0. ROS를 사용하는 경우 심한 delay 발생...
