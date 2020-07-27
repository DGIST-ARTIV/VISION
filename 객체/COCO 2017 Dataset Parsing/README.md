# COCO 2017 Dataset Parsing for yolov3
Author: 이  구
date: 2020.07.27

### How to use
* cocotoyoloconverter.py   
 
   ```(python3)
   cat = {class name}
   ```
   을 수정하여 추출할 class를 선택할 수 있다. 오직 하나의 class만 추출할 수 있다.

   ```(python3)
   mystring = str("1 " + str(truncate(x, 7)) + " " + str(truncate(y, 7)) + " " + str(truncate(w, 7)) + " " + str(truncate(h, 7)))
   ```
   1을 수정하여 class num을 바꿔줄 수 있다.   

* compare.py   
cocotoyoloconverter.py를 실행하면 해당하는 class의 txt 파일만 생성된다. compare.py는 이 텍스트 파일에 대응되는 이미지 파일을 저장하는 코드이다.

* listing.py   
해당 폴더 내에 있는 jpg 형식 파일의 리스트를 만들어준다.    

* result_listing.py     
listing 된 txt 파일들을 다시 한번 합쳐준다. yolov3 학습 시 train.txt에 해당하는 파일을 생성해준다.      

* convert_class_num_for_multi_object.py   
폴더별로 데이터를 정리해 놓은 경우, 해당 폴더 안에 있는 이미지의 모든 객체의 클래스 넘버를 바꿔준다.


