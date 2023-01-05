annotation에도 포맷이 존재한다. 이를 구분해야 하는 이유는 annotation이 어떻게 저장되어 있는지를 알아야 코드를 짜는데 편하기 때문이다.

1. COCO format
2. Pascal VOC format
3. KITTI format
4. Cityscapes dataset
5. BDD100K dataset


# 1. COCO format

object detection, segmentation에 가장 많이 사용되는 포맷으로서 data.json과 같은 파일이 존재하고, 이 json파일에 각 이미지의 annotation 정보가 들어있다.

```txt
{
  "info" : {...},
  "licenses" : [...],
  "images" : [...],
  "categories" : [...],
  "annotations" : [...],
}
```

- info : discription, verson, contributor 등 이미지의 high level 정보
- licenses
- images : 데이터셋의 전체 이미지 이름, 각각의 width, height, id
- categories : 탐지 유무, id, 객체 이름
- annotations : 이미지 id, 카테고리, 박스 위치 등 자세한 레이블 정보

&nbsp;

크게는 위와 같이 분류되어 있다. 여기서 가장 중요한 부분인 `images`, `categories`, `annotations`은 다음과 같다. `info`나 `licenses`는 코드를 구성할 때는 불필요하다.

&nbsp;

```txt
"images": 
[
  {
    "license": 1,
    "file_name": "000000324158.jpg",
    "coco_url": "http://images.cocodataset.org/val2017/000000324158.jpg",
    "height": 334,
    "width": 500,
    "date_captured": "2013-11-19 23:54:06",
    "flickr_url": "http://farm1.staticflickr.com/169/417836491_5bf8762150_z.jpg",
    "id": 324158
  },
  ...
],
```

- license
- file name : 이미지명 
- coco url : 이미지 url
- height : image height
- width : image width
- date captured : time for captured
- filckr url
- id : annotations->image-id 와 매칭

&nbsp;

```txt
"categories": 
[  
  {
    "supercategory": "person", 
    "id": 1,
    "name": "person"
    "keypoints": [
        "nose","left_eye","right_eye","left_ear","right_ear",
        "left_shoulder","right_shoulder","left_elbow","right_elbow",
        "left_wrist","right_wrist","left_hip","right_hip",
        "left_knee","right_knee","left_ankle","right_ankle"
    ],
    "skeleton": [
        [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],
        [6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]
    ],
  },
  {
    "supercategory": "vehicle",
    "id": 2,
    "name": "bicycle",
    "keypoints" : [...],
    "skeleton" : [...],
  },
  {
    "supercategory": "vehicle",
    "id": 3,
    "name": "car",
    "keypoints" : [...],
    "skeleton" : [...],
  },
    {
    "supercategory": "vehicle",
    "id": 4,
    "name": "motocycle",
    "keypoints" : [...],
    "skeleton" : [...],
  },
    {
    "supercategory": "vehicle",
    "id": 5,
    "name": "airplane",
    "keypoints" : [...],
    "skeleton" : [...],
  },
  {
    "supercategory": "vehicle",
    "id": 6,
    "name": "bus",
    "keypoints" : [...],
    "skeleton" : [...],
  },
  ...
],
```

- supercategory : 그룹
- id : 객체 id
- name : 객체명
- keypoints : keypoint 위치
- skeleton : keypoint간의 연결 고리. e.g. [16,14] -> left ankle과 left knee가 연결되어 있다.

&nbsp;

```txt
"annotations" :
[
  {
    "image_id" : 0, 
    "category_id" : 6, 
    "segmentation" : [[173,61], [173,62], ...] 
    "area" : 606046 
    "bbox" : [831.0, 619.0, 1379.0, 764.0]
    "iscrowd" : 0,
    "id": 0,
    "keypoints" : [229,256,2,...,],
    "caption" : "A person riding a very tall bike in the street",
  },
  ... 
]
```

- image_id : images에 있는 id와 매칭
- category_id : categories에 있는 id와 매칭
- segmentation : segmentation 영역
- area : 영역 넓이
- bbox : bbox x좌표, y좌표, 너비, 높이
- iscrowd : 가려짐 정도. 0 or 1, 0이면 x, 1이면 가려짐.
- id : annotation id.
- keypoints : x,y,v  픽셀 위치 x,y와 visibility v (0:not labeled or 1: labeled but not visible or 2: labeled and visible)
- caption : caption 내용

&nbsp;

## reference

1. https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch
2. https://ukayzm.github.io/cocodataset/
3. https://velog.io/@zeen263/%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-3-Segmentation-1.-EDA

&nbsp;

&nbsp;

# 2. Pascal VOC format

voc의 경우 각 이미지의 annotation 정보가 들어 있는 파일이 각각 존재한다. 예를 들어 00001.jpg의 annotation은 00001.txt와 같은 annotation 파일에 있다. txt또는 xml로 존재한다.

또한, voc는 이미지가 들어있는 `jpegimages` 폴더, annotation 정보가 들어 있는 `annotations` 폴더, 이미지 리스트가 저장되어 있는 `imagesets` 폴더가 있다. 추가로 segmentation에 대한 정보 폴더인 `segmentationclass`와 `segmentationobject` 폴더도 있다. 각각 전자는 semantic segmentation을 위한 label 이미지이고, 후자는 instance segmentation을 위한 label 이미지이다.

```txt
voc20xx
├── Annotations
    ├── a.jpg
    ├── b.jpg
    ├── c.jpg
├── ImageSets
    ├── a.xml
    ├── b.xml
    ├── c.xml
├── JPEGImages
    ├── train.txt
    ├── eval.txt
    ├── test.txt
├── SegmentationClass
├── SegmentationObject
```

&nbsp;

xml안에는 다음과 같은 구조로 되어 있다.

```xml
<annotation>
    <folder>VOC2007</folder>
    <filename>000001.jpg</filename>
    <source>
        <database>The VOC2007 Database</database>
        <annotation>PASCAL VOC2007</annotation>
        <image>flickr</image>
        <flickrid>341012865</flickrid>
    </source>
    <owner>
        <flickrid>Fried Camels</flickrid>
        <name>Jinky the Fruit Bat</name>
    </owner>
    <size>
        <width>353</width>
        <height>500</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>dog</name>
        <pose>Left</pose>
        <truncated>1</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>48</xmin>
            <ymin>240</ymin>
            <xmax>195</xmax>
            <ymax>371</ymax>
        </bndbox>
    </object>
    <object>
        <name>person</name>
        <pose>Left</pose>
        <truncated>1</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>8</xmin>
            <ymin>12</ymin>
            <xmax>352</xmax>
            <ymax>498</ymax>
        </bndbox>
    </object>
</annotation>
```

- folder
- filename : 이미지 이름
- source
- owner
- size
    - width : 각 이미지 width
    - height : 각 이미지 height
    - depth : 각 이미지 channel
- object
    - name : 클래스 이름
    - bndbox : 
        - xmin : 좌상단 x
        - ymin : 좌상단 y
        - xmax : 우하단 x
        - ymax : 우하단 y

&nbsp;

## reference

1. https://deepbaksuvision.github.io/Modu_ObjectDetection/posts/02_01_PASCAL_VOC.html

&nbsp;

&nbsp;

# 3. KITTI format

kitti object detection label은 다음과 같다.

```txt
Truck 0.00 0 -1.57 599.41 156.40 629.75 189.25 2.85 2.63 12.34 0.47 1.49 69.44 -1.56
```

<img src="https://dkssud8150.github.io/assets/img/dev/week10/day5/kitti.png">

- type: 총 9개의 클래스에 대한 정보이다. tram은 길거리에 있는 기차를 말하고, misc는 구별하기 어려운 애매한 차량들을 지칭하는 것이고, doncares는 너무 멀리 있어서 점으로 보이거나, 잘려서 보이는 차량들에 대한 클래스로 지정해놓은 것이다.
- truncated : 차량이 이미지에서 벗어난 정도를 나타낸다. 0~1사이의 값으로 표현된다.
- occluded : 다른 오브젝트에 의해 가려진 정도를 나타낸다. 0은 전체가 다 보이는 것, 1은 일부만 ,2는 많이 가려진 정도, 3은 아예 안보이는 것으로 나타낸다.
- alpha : 객체의 각도를 나타낸다.
- bbox : left, top, right, bottom에 대한 bounding box 좌표를 나타낸다.

&nbsp;

## reference

1. https://dkssud8150.github.io/posts/yolo/#prepare-data

&nbsp;

&nbsp;

# 4. Cityscapes dataset

```

cityscapes
├── annotations
├── gtFine
│   ├── test
│       ├── berlin
│       ├── bielefeld
│       ├── ...   
│   ├── train
│       ├── ...
│   └── val
│       ├── ...
└── leftImg8bit
│   ├── test
│       ├── berlin
│       ├── bielefeld
│       ├── ...   
│   ├── train
│       ├── ...
│   └── val
│       ├── ...

```

gtFine에는 50개의 도시별로 annotation이미지와 각 클래스별 polygon 정보가 들어 있다.

```txt
{
    "imgHeight": 1024, 
    "imgWidth": 2048, 
    "objects": [
        {
            "label": "road", 
            "polygon": [
                [
                    0,769
                ],
                [
                    290,574
                ],
                ...
            ]
        },
        {
            "label": "sidewalk",
            "polygon": [
                [
                    2047,532
                ],
                ...
            ]
        },
        ...
```

&nbsp;

그리고 `leftImg8bit`에는 스테레오 카메라 중 왼쪽 카메라의 이미지들이 8bit 즉, LDR 형태로 저장되어 있다. DR, dynamic range(동적 범위)는 사진에서 밝기가 가장 밝은 부분과 가장 어두운 부분의 밝기의 비율을 나타낸다. 16bit는 8bit보다 더 큰 비율을 가진 표현 방식으로 HDR(High Dynamic Range) 라 한다.

&nbsp;

- Class 종류

class는 총 30개가 있다.

| Group |	Classes |
| --- | --- |
| flat |	road · sidewalk · parking+ · rail track+ | 
| human |	person* · rider* |
| vehicle |	car* · truck* · bus* · on rails* · motorcycle* · bicycle* · caravan*+ · trailer*+ |
| construction |	building · wall · fence · guard rail+ · bridge+ · tunnel+ |
| object |	pole · pole group+ · traffic sign · traffic light | 
| nature |	vegetation · terrain | 
| sky |  sky |
| void |	ground+ · dynamic+ · static+ | 

&nbsp;

## reference

1. https://github.com/mcordts/cityscapesScripts
2. https://www.vbflash.net/65

&nbsp;

&nbsp;

# 5. BDD100K dataset

<img src="/assets/bdd100k_dataset.png">

[데이터셋](https://bdd-data.berkeley.edu/portal.html#download)에 가보면 여러 가지가 있다. 여기서 100k images는 객체 탐지, 이동 가능 영역 탐지, 차선 탐지 등에 사용되는 데이터셋이라고 한다. 그리고 10k images는 segmantic, instance, panoptic segmentation에 대한 데이터셋이라고 한다. 용량은 100k는 5.3GB, 10k는 1.1GB이다.

annotation은 `label`탭에 있다. panoptic segmentation에 대한 폴더 구조는 다음과 같다.

```txt
- labels
    - pan_seg
        - bitmasks
            - train
            - val
        - colormaps
            - train
            - val
        - polygons
            - pan_seg_train.json
            - pan_seg_val.json
```

- bitmasks : 각 이미지에서 객체 유뮤에 대한 라벨의 정보를 8bit(1 channels)이미지로 저장되어 있다. 각 픽셀에는 카테고리가 저장되어 있다. 255는 ignore 속성
- colormaps : 각 이미지의 라벨들은 RGBA png 파일로 저장되어 있다.
  - R : 카테고리 ID로 사용되고, 0~1의 값을 가진다. 0은 배경
  - G : instance 속성
    - truncated  
    - occluded
    - crowd
    - ignore 
    - `G = (truncated << 3) + (occluded << 2) + (crowd << 1) + ignore`
  - B : instance segm에 사용될 annotation id.
  - A : segmentation tracking에 사용
    - `value = (B << 8)  + A`
  <img src="/assets/pano_seg_color.png">


```txt
[
  {
    "name": "",
    "timestamp": "",
    "labels": [
      "id": "",
      "category": "sky",
      "poly2d": [
        "vertics": [[...], [...], [...], [...]]
      ],
      "types": "LLLL",
      "closed": false,
    ],
  }
]
```

- polygons : bezier curve 로서 표현되어 있다.
  - name : anno image 이름
  - timestamp
  - label
    - id : 영역 id
    - category : 영역 카테고리
    - poly2d : [bezier curve](https://blog.coderifleman.com/2016/12/30/bezier-curves/) 형태로 저장되어 있다.
      - vertices(control points) : 영역 정점(꼭지점)
    - types : 정점을 `L`로 표현하여 L의 개수만큼 정점 존재
    - closed : true/false

## reference

1. https://doc.bdd100k.com/download.html
2. https://blog.coderifleman.com/2016/12/30/bezier-curves/
