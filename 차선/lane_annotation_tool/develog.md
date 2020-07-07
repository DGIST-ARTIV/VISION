# Develog
> Author : Eunbin Seo <br/>
> Date : 2020.06.24. ~ 2020.06.26.

## ENet-SAD를 위한 annotation tool을 이용하기로 했다.  

#### labelme라는 annotation tool을 이용하고자 했다.  
[labelme](https://github.com/wkentaro/labelme)를 다운받자.  
``` bash
sudo apt-get install python3-pyqt5  # PyQt5
sudo pip3 isudo pip3 install labelmenstall labelme
```
그리고 실행시켜 json 파일을 얻어 segmentation된 이미지 파일을 얻을 수 있다.  
![labelme](https://user-images.githubusercontent.com/53460541/86797732-fd029680-c0aa-11ea-91f7-86c38c48ec7a.png)  

##### 하지만, 우리는 CULane dataset 형식을 따라야했다...

#### 직접 annotation tool을 만들어보자!
우리가 구현해야할 기능으로는 자잘자잘한 것을 빼면 총 6가지가 있다.
1. 마우스 이벤트를 통해 점을 찍는 곳이 잘 보이게 점을 그려주는 기능
2. 차선 4개를 구분해서 data를 담아두는 기능
3. 한 프레임당 담아둔 data를 x, y 좌표 순으로 txt 파일로 저장하는 기능
4. segmentation png 파일을 만들어 저장하는 기능
5. legend 그리기
6. annotation한 모든 이미지 파일과 txt 파일 목록을 가지고 있는 txt 파일로 저장하는 기능
이틀에 걸쳐 만들었고, 그 이후에 사용하면서 코드를 고쳐나갔다.
사용방법과 코드는 같은 파일 내에 적어두었다. [link](https://github.com/DGIST-ARTIV/VISION/tree/master/%EC%B0%A8%EC%84%A0/lane_annotation_tool)

#### 수정한 내용
1. culane seg label에 대해 두께가 같은지 확인 필요.  
논문을 쓴 저자가 이용한 [seg_label_generate](https://github.com/XingangPan/seg_label_generate)를 확인해본 결과 두께가 16으로 같았다.(~~놀라움!!!!~~)  
그러나 seg label을 만드는 방식이 완전히 달라서 수정하고자 했다. dashed line과 solid line도 구분하려고 weight를 따로 주었지만 우리는 필요가 없어 polyfit으로만 부드럽게 하고자 했다.  
코드는 다음과 같다.  
``` python
def draw_polynomial_regression_lane(line, seg, counter):
	x_list = []
	y_list = []
	for idx, pts in enumerate(line):
		x, y = int(pts[0]), int(pts[1])
		x_list.append(x)
		y_list.append(y)
	try:
		fp1 = np.polyfit(np.array(x_list), np.array(y_list) , 2)
		f1 = np.poly1d(fp1)
		y_list = np.polyval(f1, x_list)
		draw_poly = np.array([list(zip(x_list, y_list))], np.int32)
		seg=cv2.polylines(seg, np.asarray(draw_poly), False , counter+1, 16)
	except:
		pass
	return seg
  ```
그러나 결과가 좋지 않아서 원래하던 방식으로 진행하고자 했다.   
![best_seg_is_simple_line](https://user-images.githubusercontent.com/53460541/86800523-135e2180-c0ae-11ea-8c9d-a26a174f93c0.png)  

2. n으로 넘어가는 frame 수가 맞지 않는다??!  
코드를 어디서 잘못 짰는지 잘 모르겠지만 일단 두 배 차이가 나서 cap.read를 한 번씩 더해주었다. 그 결과 비슷한 위치로 이동했지만, 완전히 같은 frame으로 이동하지 못했다.  
일단, annotation만 잘하면 되니 이런 issue는 넘어가기로 했다.
