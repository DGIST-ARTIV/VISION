# Ros camera calibration
> reference: http://wiki.ros.org/camera_calibration/Tutorials/MonocularCalibration

### setting
1. intstall uvc-camera pkg or usb_cam pkg
~~~(bash)
sudo apt-get install ros-melodic-uvc-camera
sudo apt-get install ros-melodic-image-*
sudo apt-get install ros-melodic-image-view
~~~
내 컴퓨터는 usb_cam pkg를 이용했다.
2. OpenCV camera calibration
OpenCV를 설치하자. 다음 링크를 참고한다. https://sunkyoo.github.io/opencv4cvml/OpenCV4Linux.html
3. camera가 잘 연결되는지 확인해보자.
~~~(bash)
lsusb
ls -ltr /dev/video*
~~~
잘 연결되었을 시에 다음과 같은 메시지가 출력될 것이다.  
![check_camera](https://user-images.githubusercontent.com/53460541/86659449-d437be00-c024-11ea-83fe-51e43fed1315.PNG)

### Run!
1. ros1을 키고, roscore를 켜준다.(모두가 알죠? source /opt/ros/melodic/setup.bash 모르면 이 [링크](/ROS/readMe.md)타고 들어가서 공부해오세요.)
2. 또 다른 창에 ros1을 키고, 다음 명령어 입력
~~~(bash)
rosrun uvc_camera uvc_camera_node 
~~~
or
~~~(bash)
rosrun usb_cam usb_cam_node
~~~

3. 노드를 실행시키면 camera calibration file이 없다는 내용으로 warning이 뜬다. camera.yaml 파일을 생성하자  
![warning](https://user-images.githubusercontent.com/53460541/86659475-d7cb4500-c024-11ea-9a88-82859b2ba8ca.PNG)

4. 3번까지 진행한 후, 확인해야하는 항목이 있다.
~~~(bash)
rostopic list
~~~
후 /camera_info가 /camera 하위에 있는지 / 하위에 있는지 확인해야한다. 다음과 같이 나오면 된다.  
![rostopic_list](https://user-images.githubusercontent.com/53460541/86659469-d69a1800-c024-11ea-9e9b-3d4821b719a9.PNG)

5. camera calibration을 실행시키자. 샘플 데이터를 얻어오자.
~~~(bash)
rosrun camera_calibration cameracalibrator.py --size 8x6 --square 0.024 image:=/usb_cam/image_raw camera:=/
~~~
4번에서 만약 /camera/camera_info로 나온다면, 마지막 항목을 camera:=/camera로 수정해야한다.
~~~(bash)
rosrun camera_calibration cameracalibrator.py --size 8x6 --square 0.024 image:=/usb_cam/image_raw camera:=/
~~~
잘 실행된다면, 다음과 같이 sample data를 얻을 수 있을 것이다.  
![sample_data](https://user-images.githubusercontent.com/53460541/86659471-d732ae80-c024-11ea-8427-b05c3d1d74a1.PNG)  
sample은 41개 이상 뽑아야 위 사진처럼 calibrate button이 활성화된다.

6. calibrate button을 누르고 다 끝나면 save button을 눌러 저장한다.
calibration한 결과는 밑의 사진과 같이 나온다.
![calibrate](https://user-images.githubusercontent.com/53460541/86659486-dac63580-c024-11ea-88cc-7315f82d5d26.PNG)  
save button을 누르면, 파일들이 /tmp/calibrationdata.tar.gz로 저장되었다는 문구를 볼 수 있다.

7. /tmp/calibrationdata.tar.gz 파일을 새로운 파일 ~/camera_calibration_data 안에 압축을 풀어준다.
~~~(bash)
cd tmp
ls
tar -xzvf calibrationdata.tar.gz -C ~/camera_calibration_data/
~~~
![extract](https://user-images.githubusercontent.com/53460541/86659458-d568eb00-c024-11ea-8ea4-facd1938911f.PNG)  
파일 압축을 풀어주면 camera_calibration_data 파일은 다음과 같이 보여진다.  
![file](https://user-images.githubusercontent.com/53460541/86659462-d6018180-c024-11ea-8586-b0a8b37d2877.PNG)

8. ost.txt 를 yaml 로 바꾸어주고 최종 yaml 파일을 확인해본다.
~~~(bash)
mv ost.txt ost.ini
rosrun camera_calibration_parsers convert ost.ini camera.yaml  # ost.txt to yml
mkdir ~/.ros/camera_info  # for saving our calibration yaml
mv camera.yaml ~/.ros.camera_info  # move the yaml file
~~~
ost.yaml 파일을 열어보면 밑의 화면과 같이 나올 것이다.  
![final](https://user-images.githubusercontent.com/53460541/86659466-d69a1800-c024-11ea-8efa-ed55a409a1f1.PNG)
