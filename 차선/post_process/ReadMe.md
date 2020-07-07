# Develog
> Author : Eunbin Seo <br/>
> Date : 2020.06.27. ~ 2020.07.01.

## Probline
ENet-SAD가 50 frame이 나오는 것을 확인했고, 이제 차선이 있을 것 같다는 probline 그려 표시하기로 했다.  
ENet-SAD가 100% 정확하게 차선이 있다는 것을 알아맞추는 것이 아니기 때문에 outlier들이 많았다.  
-> outlier 처리를 통한 성능 향상  
-> 원내 dataset을 학습을 통한 성능 향상  
두 가지를 목표로 5일간 probline과 annotation을 진행했다. 여기서는 probline에 대한 이야기만 나눌 예정이다.

4개의 차선 즉, leftleft, left, right, rightright 차선을 그려주기로 했다. 총 5차례의 시도에 거쳐 어느 정도 안정화시켰다.  
#### 1. ENet-SAD에서 나온 probability list의 점들을 모두 이어보았다.  
![draw1](https://user-images.githubusercontent.com/53460541/86770711-11836680-c08c-11ea-96ba-70c0a6a2e870.gif)  
결과는 당연히 좋지 않았다. 처음엔 probline이니까 차선이 있을 것 같은 점들을 모두 이으면 된다고 생각했는데, outlier가 있음을 간과했다.  
코드의 일부는 다음과 같다. 
``` python
def draw_lane_simple_method(line, img):
  for pts in range(len(line)):
    if line != [None] and pts != len(line)-1:
      img = cv2.line(img,tuple(line[pts]),tuple(line[pts+1]),(pts*50,0,0),2)
  return img
```

#### 2. linear regression을 통해 outlier를 처리해보았다.  
![draw2_1](https://user-images.githubusercontent.com/53460541/86770710-10ead000-c08c-11ea-9b92-efa07af69d05.gif)  
outlier를 모두 무시했으나 두 가지 문제점이 있다.  
- linear regression이다 보니 곡선이 나왔을 경우 처리하지 못한다.  
- outlier가 주를 이룰 때에도 선을 그려 엉망진창이 된다.  
코드의 일부는 다음과 같다.
``` python
def draw_linear_regression_lane(line, img):
  global count, colorlist
  lane_img = np.zeros_like(img)  
  points =[]
  x_list = []
  y_list = []
  if line != [None]:
    for idx, pts in enumerate(line):
      if 50<int(pts[0])<750 and 160 < int(pts[1]) < 260:
        x, y = int(pts[0]), int(pts[1])
        points.append((x,y))
        x_list.append(x)
        y_list.append(y)
  # linear regression
  vx, vy, x, y = cv2.fitLine(np.array(points), cv2.DIST_L2, 0, 0.11, 0.11)
  line = [float(vx),float(vy),float(x),float(y)]
  left_pt = int((-x*vy/vx) + y)
  right_pt = int(((lane_img.shape[1]-x)*vy/vx)+y)
  lane_img = cv2.line(lane_img,(lane_img.shape[1]-1,right_pt),(0,left_pt),255,2)
  img = cv2.addWeighted(src1=lane_img, alpha=1, src2=img, beta=1., gamma=0.)
  return img
  ```
--> 코드가 조금 많이 드럽다. cv2.fitLine이 먹는 array를 만들기 위해 x와 y의 좌표를 따로 나누어 array형식으로 전환후 line을 그렸다.

#### 3. 곡선의 문제를 해결하기 위해 polynomial regression을 해보자.  
![draw3](https://user-images.githubusercontent.com/53460541/86770715-12b49380-c08c-11ea-984d-b32d3c5d0450.gif)  
결과가 나쁘진 않았다. 하지만 여전히 outlier를 해결하지 못하는 결과를 볼 수 있다.  
이를 해결하기 위해서는 머리를 조금 더 써야할 것 같다.  
코드의 일부는 다음과 같다.  
``` python
def draw_polynomial_regression_lane(line, img):
  #(linear regression과 동일해 생략)
  
  # polynomial regression
  fp1 = np.polyfit(np.array(x_list), np.array(y_list) , 2)
  f1 = np.poly1d(fp1)
  y_list = np.polyval(f1, x_list)
  draw_poly = np.array([list(zip(x_list, y_list))], np.int32)
  print(np.asarray(draw_poly))
  lane_img=cv2.polylines(lane_img, np.asarray(draw_poly), False, (255,255,255), 5)
  img = cv2.addWeighted(src1=lane_img, alpha=1, src2=img, beta=1., gamma=0.)
  return img
  ```
  
#### 4. outlier를 처리하기 위해 ransac을 적용시켰다.  
![draw4](https://user-images.githubusercontent.com/53460541/86770717-13e5c080-c08c-11ea-92e2-7ac5ff6db89a.gif)  
ransac은 random sample consensus의 약자로, 가장 많은 수의 데이터들로부터 지지받는 모델을 선택한다. least square method보다 더 효과적인 모델을 만들어내기도한다.  
하지만 우리가 처리하는 data들의 양이 많지 않다. 가장 많은 수가 어떤 값을 가리키고 있는지 명확하지 않다.  
inlier가 주를 이루고 있을 때가 존재해 ransac으로 몇몇 점들은 무시하지만 outlier가 주를 이루고 있을 땐 inlier를 outlier로 취급해버린다.  
또한, for문을 돌면서 모델을 만들 sample들을 선택하기 때문에 시간도 오래걸린다. 조작해야하는 parameter 수가 많은 것도 문제였다.  
그 결과로, 좋은 성능을 이끌어내기 어려웠다. 더 좋은 성능을 내는 알고리즘을 고민해봐야한다.  
코드의 일부는 다음과 같다.  
``` python
def draw_lane_ransac(line, img):
  #(linear regression과 동일해 생략)
  for idx, pts in enumerate(line):
    if line != [None] and 50<int(pts[0])<750 and 160 < int(pts[1]) < 260:
      x, y = int(pts[0]), int(pts[1])
      cv2.circle(img, (x,y), 10, colorlist[count-1], -1)
      cv2.putText(img, str(idx), (x, y),cv2.FONT_HERSHEY_COMPLEX,0.4,(255,255,255),1)
      points.append((x,y))
      x_list.append(x)
      y_list.append(y)

  lane_img = ransac_polyfit(np.array(x_list), np.array(y_list),lane_img)
  img = cv2.addWeighted(src1=lane_img, alpha=1, src2=img, beta=1., gamma=0.)
  count +=1
  return img
  ```

#### 5. start point를 잡고 각도 범위 밖이면 삭제하는 알고리즘 적용시켰다.  
![draw5](https://user-images.githubusercontent.com/53460541/86770720-1516ed80-c08c-11ea-8d3e-da22733bf592.gif)  
위의 영상들과 달리 안정적인 것을 볼 수 있다. 각 차선마다 나올 수 있는 각도를 계산해 각도의 범위로 outlier를 제거했다.  
코드의 일부는 다음과 같다.  
``` python
def using_degree(line, img, start_point, idx):
  degree = [[-85, -75], [-70, -60], [45, 55], [70, 80]]
  print(degree[idx][1])
  x1, y1 = start_point[0], start_point[1]
  x_new = [start_point[0]]
  y_new = [start_point[1]]
  for pts in line:
    if 50 <int(pts[0])<750 and 160 < int(pts[1]) < 280:
      x2, y2 = int(pts[0]), int(pts[1])
      cv2.circle(img, (x2,y2), 10, colorlist[count-1], -1)
      cv2.putText(img, str(idx), (x2, y2),cv2.FONT_HERSHEY_COMPLEX,0.4,(255,255,255),1)
      #slope_degree = for_degree(x1,x2, y1, y2)
      slope_degree = (np.arctan2(x1-x2, y1-y2) * 180) / np.pi
      if degree[idx][0] < (slope_degree) < degree[idx][1]:
        x_new.append(x2)
        y_new.append(y2)
  return x_new, y_new

def draw_lanes(lanes, img):
  lane_img = np.zeros_like(img)  
  start_points = [[0, 220], [110, 286], [580,287], [798, 240]]
  for idx, line in enumerate(lanes):
    if line == [None]:
      pass
    else:
      x_list, y_list = using_degree(line, img, start_points[idx], idx)
      try:
        fp1 = np.polyfit(np.array(x_list), np.array(y_list) , 2)
        f1 = np.poly1d(fp1)
        y_list = np.polyval(f1, x_list)
        draw_poly = np.array([list(zip(x_list, y_list))], np.int32)
        print(np.asarray(draw_poly))
        lane_img=cv2.polylines(lane_img, np.asarray(draw_poly), False, (255,255,255),5)
        img = cv2.addWeighted(src1=lane_img, alpha=1, src2=img, beta=1., gamma=0.)
      except:
        pass
  return lane_img, img
  ```
--> 개선시켜야할 점은 여전히 존재한다.
- 차가 가운데로 항상 다니는 것이 아니기 때문에 start point가 항상 바뀐다. start point 초기값을 어떻게 잡을지 또 다른 알고리즘을 통해 결정해야한다.
- 이 방법을 통해 line을 그릴 시에 0.01초 정도 걸린다. 계산 가속화/최적화를 하기 위한 방법을 모색해야한다.

## 계산 가속화 및 최적화
numba를 이용해 계산 가속화를 하려고 했다. 이중 list 처리를 하지 못하고 numba를 왜 도대체 어디에 이용하는지 의문을 가졌다.  
지원하는 library가 적고 심지어 numpy library 전체를 지원하지 않아 쓰기가 까다롭다. 다른 방식을 찾아보아
