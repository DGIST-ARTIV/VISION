# Develog
> Author : Eunbin Seo <br/>
> Date : 2020.06.30 ~ 2020.07.05.

## Dataset
직접 만든 [annotation tool](https://github.com/DGIST-ARTIV/VISION/tree/master/%EC%B0%A8%EC%84%A0/lane_annotation_tool)을 이용했고 잘 학습되는지 14frame 정도로 test해보았다.  
잘 학습되는 것을 보았고 성능향상을 위해 annotation을 많이 하는 것을 필요로 했다.  
이틀에 걸쳐 현진이과 같이 낮 영상과 밤 영상에서 총 1000장 정도의 frame을 annotation했다.  
Culane dataset 형태에 맞춰 culane dataset과 합쳐 ENet-SAD train을 시켰다.  
다음은 culane dataset만을 이용해 test한 결과이다.  
![only_culane-_online-video-cutter com_](https://user-images.githubusercontent.com/53460541/86783514-5ebb0480-c09b-11ea-987d-0871040c6a19.gif)  
다음은 우리가 annotation한 dataset을 포함시킨 결과이다.
![culane_and_ours-_online-video-cutter com_](https://user-images.githubusercontent.com/53460541/86783519-61b5f500-c09b-11ea-92e5-c876b145ea8c.gif)  
아직까지 큰 차이는 없어보이지만 더 많은 dataset을 만들게 되면 더 좋은 성능을 가지게 될 것이라고 믿고 있다. 

## 학습전략
- annotation tool을 시나리오와 함께 팀원들에게 배포해 많은 dataset을 얻어낸다.  
- 여러 상황에 대한 video(아침, 점심, 밤, 언덕, 내리막길 등에 대한)을 취득할 것이다.  
- augmentation을 통해 dataset을 늘려줄 것이다.

## Augmentation
#### 1. cv2 기능을 이용해 frame 밝기 조절, blur 처리 등 해보기
- 밝기 조절: 이미지의 각 픽셀에 값을 더하고 빼는 것으로 밝기 조절 가능
``` python
  # 밝게
  M = np.ones(img.shape, dtype="uint8") * 50
  added = cv2.add(img, M)
  cv2.imshow("Added", added)
  cv2.waitKey(0)

  # 어둡게
  M = np.ones(img.shape, dtype="uint8") * 70
  subtracted = cv2.subtract(img, M)
  cv2.imshow("Subtracted", subtracted)
  cv2.waitKey(0)
  ```
- blur : cv2의 blur 자체 기능 이용하여 blur 처리 가능
``` python
    blur = cv2.bilateralFilter(img,9,130,130) #bilateral filter, default = (9, 75, 75)
    #blur2 = cv2.medianBlur(img, 13) #미디안 블러링
    #blur3 = cv2.GaussianBlur(img, (13,13), 0) #가우시안 블러링
    #blur4 = cv2.blur(img, (13,13)) #평균 블러링, 커널 크기가 5*5
```

#### 2. imgaug
##### - imgaug 설치하기
 ``` bash
 pip3 install six numpy scipy Pillow matplotlib scikit-image opencv-python imageio Shapely
 ```
 imgaug를 이용하기 위해서는 Shapely가 필수적이다. Shapely가 import가 되지 않는다면, sudo apt-get 을 이용하여 설치해보자.
 ``` bash
sudo apt-get update
sudo apt-get install python-shapely
```
Shapely가 잘 다운로드된 것을 확인하고, imgaug 모듈을 설치하자.
``` bash
pip3 install imgaug
````
##### - imgaug 사용하기
augmentation은 차선 인식뿐만 아니라 객체 인식을 위한 dataset에 적용시키기 때문에 augmentation을 위한 library를 이용하는 것이 효율적이다.  
imgaug는 keypoint나 landmarks도 같이 augmentation이 가능하기 때문에 객체 dataset에서 사용하기 좋다.  
차선 인식을 하는 ENet-SAD는 pytorch를 프레임워크로 사용하기 때문에 pytorch와 연동하는 방법을 알아보자.  
[참고자료](https://colab.research.google.com/drive/109vu3F1LTzD1gdVV6cho9fKGx7lzbFll)  
``` python
class ImgAugTransform:
  def __init__(self):
    self.aug = iaa.Sequential([
        iaa.Scale((224, 224)),
        iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
        iaa.Fliplr(0.5),
        iaa.Affine(rotate=(-20, 20), mode='symmetric'),
        iaa.Sometimes(0.25,
                      iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
                                 iaa.CoarseDropout(0.1, size_percent=0.5)])),
        iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
    ])
      
  def __call__(self, img):
    img = np.array(img)
    return self.aug.augment_image(img)

transforms = ImgAugTransform()

dataset = torchvision.datasets.ImageFolder('pytorch-examples/data/', transform=transforms)

# 기존 torch의 transforms 함수와 같이 사용하고 싶다!
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([# transforms.RandomRotation(20),
                                aug_transforms,
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

# loader 만들기
# Download and load the training data
trainset = datasets.CIFAR10('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.CIFAR10('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
```

#### 3. albumentation
augmentation 검색을 하다가 2020년 3월에 나온 augmentation library인 albumentation을 알아버렸다!!  
2.에서 사용하려고 했던 imgaug보다도 처리하는 속도가 훨씬 빠르다.  
참고자료는 [여기서](https://colab.research.google.com/drive/1JuZ23u0C0gx93kV0oJ8Mq0B6CBYhPLXy#scrollTo=GwFN-In3iagp&forceEdit=true&offline=true&sandboxMode=true),
[저기서](https://github.com/albumentations-team/albumentations) 그리고 [여기](https://hoya012.github.io/blog/albumentation_tutorial/)에서 확인가능하다.  
##### - albumentation 설치하기
간단히 pip3 로 설치가 가능하다.
``` bash
pip3 install albumentations
```
##### - albumentation 사용하기
albumentation 기능 중 날씨를 변화 시켜주는 기능이 있다. 날씨 변화에 민감한 자율주행차량에 큰 도움을 줄 수 있을 것 같다.  
그 외에도 flip, rotation, blur 등 많은 기능이 있다.  
``` python
import numpy as np
import cv2
import albumentations as albu

def augment_and_show(aug, image, window):
    image = aug(image=image)['image']
    image = cv2.resize(image, dsize = (800,288))
    cv2.imshow(window, image)
    k = cv2.waitKey(0)
    if k == 27: # esc key
    	cv2.destroyAllWindow()

image = cv2.imread("test.png")

aug = albu.RandomRain(p=1, brightness_coefficient=0.9, drop_width=1, blur_value=5)
augment_and_show(aug, image, "rain")

aug = albu.RandomSnow(p=1, brightness_coeff=2.5, snow_point_lower=0.3, snow_point_upper=0.5)
#augment_and_show(aug, image, "snow")

aug = albu.RandomSunFlare(p=1, flare_roi=(0, 0, 1, 0.5), angle_lower=0.5)
#augment_and_show(aug, image, "sunflare")

aug = albu.RandomShadow(p=1, num_shadows_lower=1, num_shadows_upper=1, shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1))
#augment_and_show(aug, image, "shadow")

aug = albu.RandomFog(p=1, fog_coef_lower=0.3, fog_coef_upper=0.3, alpha_coef=0.1)
#augment_and_show(aug, image, "fog")
```
이를 하나하나 실행시켜보면 다음과 같은 결과가 나온다.  
![aug_result](https://user-images.githubusercontent.com/53460541/86788073-b740d080-c0a0-11ea-8742-bbd0210e8eeb.png)  

#####차선 dataset augmentation에 대한 생각 정리
- 날씨 변화에 대해서는 param 조절해서 적용 가능. (snow는 조금 더 확인을 거치고 진행)
- flip, rotation은 적용시키기 어려울 것으로 생각됨.  
만약, 좌우 반전을 시킨다고하면 seg image도 함께 적용하고 ground-truth도 바꿔주는 코드를 짜야함.  
90도 rotation을 한다면 ENet-SAD가 차선 1,2,3,4 만 골라보기 때문에 불가함.
- 색깔 변화 적용 가능. (색이 아예 다른 색으로 바뀌지 않는 선에서 적용 가능. 만약 바꾼다면 나중에 중앙선을 분리해내는 train 코드에서 어려움을 겪을듯)
- blur 적용 가능. (특히나 차가 빠르게 달릴 경우, 차선이 뿌옇게 보이는 경우도 있으므로 이런 경우를 대비해서 적용시켜보는 것도 좋을듯)

## Todo about Dataset(Augmentation)
- augmentation 종류 결정하기
- pytorch를 이용한 ENet-SAD train 코드에 바로 적용
