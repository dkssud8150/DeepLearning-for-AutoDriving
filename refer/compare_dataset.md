# 데이터셋 비교

CVPR2020에서 소개된 자율주행 오픈소스 데이터셋 TOP5를 비교하고자 한다.

&nbsp;

## 1. BDD100K (UC버클리 대학이 공개한 자율주행용 거대 데이터셋)

<img src="/assets/bdd100k.png">

BDD100K 데이터셋은 UC버클리 대학이 2018년에 공개한 자율주행 데이터셋으로 주석 처리된 10만 개 이상의 다양한 비디오 클립에 주행 장면의 가장 큰 데이터셋을 수집했다고 한다. lane detect, drivable area segmentation, object detection, segmentation, MOT, tracking 등에 대한 annotation이 존재한다.

각 비디오는 약 40초 길이이며 초당 30frame에 화질은 720p이다. 미국 전역의 거리에서 약 50,000번의 주행을 통해 수집했다고 한다. 또한 다양한 시간대와 다양한 날씨에서 촬영했다. 약 100만 대의 자동차, 30만 개 이상의 도로 표지판, 13만 명의 보행자가 포함되어 있다. 각 비디오에는 GPS, IMU 에 대한 데이터도 포함되어 있다.

&nbsp;

- site : [https://www.bdd100k.com](https://www.bdd100k.com)

&nbsp;

&nbsp;

## 2. Google Landmarks Dataset v2 (구글에서 공개한 대규모 랜드마크 데이터셋)

<img src="/assets/gldv2.png">

Google Landmarks dataset v2 데이터셋은 구글이 2019년에 공개한 랜드마크 인식 데이터셋으로, 20만 개 이상의 랜드마크에 대한 500만 개의 이미지를 제공한다.

이 데이터셋에서는 이전 데이터셋과는 달리 롱테일 현상(long-tailed class distribution), 엄청난 양의 영역 외 테스트 사진들과 클래스 내부 분산(intra-class variability) 등과 같은 실험적인 특징을 가지고 있다.

또한, 제한적인 지역에서만 촬영한 것이 아닌 세계적으로 방문하며 유명하지 않은 랜드마크에 대한 이미지들도 촬영했다.

<img src="/assets/gldv2_2.png">

&nbsp;

- site : [https://storage.googleapis.com/gld-v2/web/index.html](https://storage.googleapis.com/gld-v2/web/index.html)

&nbsp;

&nbsp;

## 3. Mapillary Street-Level Sequence (장소인식 분야의 평생학습을 위한 데이터셋)

<img src="/assets/msls.png">

Mapillary 또는 Maperial Street-Level Sequence 는 도심지와 교외의 장소 인식 분야에서의 평생학습(AI가 새로운 데잍를 학습할 때 기존에 습득한 정보를 잊는 문제를 해결하고 이전에 배운 지식과 새로운 지식을 모두 다루는 기술)을 위한 2020년에 공개된 데이터셋이다. MSLS는 총 9년에 걸쳐 수집된 데이터셋이라고 한다.

MSLS 데이터셋은 160만 개 이상의 이미지를 포함하고 있으며, 이 데이터셋은 6개 대륙, 30개 도시에서의 다양한 계절, 날씨, 일광조건, 다양한 카메라 유형, 다양한 시점, 다양한 건물 및 주변 환경, 한 장면 안에서의 움직이는 물체를 다룬다. 또한 각 이미지들에는 GPS, 촬영 시간, 나침반 각도, 낮/밤, 시야 방향(전면, 후면, 측면) 을 나타내는 메타데이터도 제공된다.

&nbsp;

- site : https://github.com/mapillary/mapillary_sls

&nbsp;

&nbsp;

## 4. nuScenes dataset (자율주행을 위한 멀티모달 데이터셋)

<img src="/assets/nuscenes.png">

nuScenes(nuTonomy scenes)는 자율주행을 위한 2019년에 공개된 대규모 공개 데이터셋이다. 

카메라 6대, 레이다 5대, 라이다 1대를 탑재한 최촤의 자율주행형 데이터셋이다. nuScenes는 보스턴과 싱가포르에서 1000개의 장면을 촬영했으며 각 장면의 길이는 20초이고, 23개 클래스와 8개 속성에 대해 3D bounding box로 labeling되어 있다. 약 140만 개의 카메라 이미지, 390만 개의 라이다 스위프(주어진 시선 방향으로 발사된 하나의 펄스에 대한 데이터), 140만 개의 레이더 스위프 및 4만 개의 키프레임(전환 시작점 및 종료점)에 있는 140만 개의 객체 bounding box를 가지고 있다. GPS, IMU 데이터도 함께 수집했다.

&nbsp;

- site : https://www.nuscenes.org/nuscenes

&nbsp;

&nbsp;

## 5. Waymo dataset (Waymo에서 공개한 자율주행 데이터셋)

<img src="/assets/waymo.png">

Waymo가 2019년에 공개한 자율주행 데이터셋으로 waymo에서 비상업적 용도로 자율주행 관련 데이터로 활용하고자 무료 제공한다. 피닉스, 마운틴뷰, 캘리포니아, 샌프란시스코 등의 25개 도시에서 1140개의 장면을 촬영했으며 각 장면의 길이는 20초이고, 5개의 고품질의 LiDAR 데이터셋을 함께 제공한다. 도심/교외 구간, 낮/밤/새벽, 일광량 정도, 비 등에 대한 다양한 조건에서 주행했다.

자동차, 보행자, 자전거, 표지 4가지만 꼼꼼하게 구분하여 label되어 있고, 전체 라벨은 120만 개의 2D label(Camera)과 1200만 개의 3D label(LiDAR)이 포함되어 있다.

센서는 미드 레인지 라이다 1개, 단거리 라이다 4개이고, 전면과 측면에 장착된 카메라 5대로 구성되어 있다. 

&nbsp;

- site : www.waymo.com/open

&nbsp;

&nbsp;

## reference

1. https://rdx-live.tistory.com/90
2. https://www.oss.kr/oss_guide/show/3dbea652-7956-45c2-9aa2-995a16649d84?page=1

&nbsp;

&nbsp;

5개 모두 자율주행 데이터셋이라곤 하지만, 실질적으로 OD(object detection), segmentation task를 수행하기 위해서는 BDD100K, nuScenes, waymo 데이터셋을 사용해야 한다.

image panoptic segmentation을 지원하는 데이터셋에는 COCO, cityscapes, kitti, bdd100k 등이 있다. lidar 데이터가 있는 데이터셋으로는 nuscenes, semanticKITTI, waymo, kitti360 등이 있다. 이 두개가 중복되는 데이터셋은 없으므로, 각각 따로 수행하는 것이 좋아보인다. 그래서 panoptic에서는 bdd100k를 사용한다. lidar에서는 nuscenes나 waymo를 사용한다.