# Learning Lightweight Lane Detection CNNs by Self Attention Distillation
[github](https://github.com/InhwanBae/ENet-SAD_Pytorch) <br/>
[sota paper](https://arxiv.org/abs/1908.00821)

Author : Eunbin Seo <br/>
Date : 2020.07.07(돌려본 건 이 날짜가 아니지만... 기억이 안남.)

github에서 문서들을 다운받고 가지고 있는 dataset으로 학습을 시켜 weight 파일을 만들어 실행시켜보았다.
결과는 다음과 같다.  
![Screenshot from 2020-06-24 22-01-31](https://user-images.githubusercontent.com/53460541/86792472-7dbe9400-c0a5-11ea-8bbf-d38b96db76c1.png)  
![Screenshot from 2020-06-24 22-02-14](https://user-images.githubusercontent.com/53460541/86792477-7eefc100-c0a5-11ea-9de3-84ffdae19d0e.png)  
차선을 인식하는 성능이 나쁘지 않게 나왔다. 

## Evaluation
|  | fps |Memory-Usage|Power(Usage/Cap)|Volatile GPU-Util|
|:--------:|:--------:|:--------:|:--------:|:--------:|
| PINet | 약 5~6 fps | 약 1400 MB | 64W/250W | 6% |
| PolyLaneNet | 약 8~9 fps | 약 8624 MB | 227W/250W | 95~100% |
| lanenet | 약 10 fps | 약 7511 MB | 100W/250W | 20% |
| ENet-SAD | 약 4~50 fps | 약 1200 MB | 64W/250W | 33% |

--> 지금까지 돌려본 network들 중 fps가 가장 좋았고 GPU 사용량도 그렇게 크지 않았다. 우리의 목표치와 가장 가까워 baseline으로 선정!  
training을 위한 차선 dataset을 만들기 위한 annotation tool을 만들고, 후처리 진행
