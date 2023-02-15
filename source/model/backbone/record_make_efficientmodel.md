# EfficientNet 및 EfficientPS 모델 구현

panoptic segmentation task를 수행하기에 앞서, 이때까지 써왔던 ResNet 대신 최신 모델이라 할 수 있는 EfficientNet 과 EfficientPS를 사용하고자 한다. 

&nbsp;

## EfficientNet 구현

### 논문 리뷰

먼저 EfficientNet에 대한 논문 리뷰는 [해당 블로그 글](https://dkssud150.github.io/post/efficientnet)을 참고하길 바란다.

원본 논문은 [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/pdf/1905.11946.pdf) 여기다.

&nbsp;

### 모델 구현

공식 EfficientNet은 tensorflow로 구현되어 있다. 그러나 이를 pytorch로 구현해준 분이 계신다. 이 깃허브 주소는 아래 참고 깃허브1을 참고하길 바란다.

먼저 모델 구조는 다음과 같다.

<img src="../../../assets/efficientnet/table1.jpg">

이를 직접 확인해보기 위한 방법으로는 torchsummary를 사용하여 더 자세하게 볼 수 있다.

```markdown
> import torchvision.models as models
> print(models.efficient_b3(pretrained=True))

Loaded pretrained weights for efficientnet-b3
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
         ZeroPad2d-1          [-1, 3, 130, 130]               0
Conv2dStaticSamePadding-2           [-1, 40, 64, 64]           1,080
       BatchNorm2d-3           [-1, 40, 64, 64]              80
MemoryEfficientSwish-4           [-1, 40, 64, 64]               0
         ZeroPad2d-5           [-1, 40, 66, 66]               0
Conv2dStaticSamePadding-6           [-1, 40, 64, 64]             360
       BatchNorm2d-7           [-1, 40, 64, 64]              80
MemoryEfficientSwish-8           [-1, 40, 64, 64]               0
          Identity-9             [-1, 40, 1, 1]               0
Conv2dStaticSamePadding-10             [-1, 10, 1, 1]             410
MemoryEfficientSwish-11             [-1, 10, 1, 1]               0
         Identity-12             [-1, 10, 1, 1]               0
Conv2dStaticSamePadding-13             [-1, 40, 1, 1]             440
         Identity-14           [-1, 40, 64, 64]               0
Conv2dStaticSamePadding-15           [-1, 24, 64, 64]             960
      BatchNorm2d-16           [-1, 24, 64, 64]              48
      MBConvBlock-17           [-1, 24, 64, 64]               0
        ZeroPad2d-18           [-1, 24, 66, 66]               0
Conv2dStaticSamePadding-19           [-1, 24, 64, 64]             216
      BatchNorm2d-20           [-1, 24, 64, 64]              48
MemoryEfficientSwish-21           [-1, 24, 64, 64]               0
         Identity-22             [-1, 24, 1, 1]               0
Conv2dStaticSamePadding-23              [-1, 6, 1, 1]             150
MemoryEfficientSwish-24              [-1, 6, 1, 1]               0
...
```

&nbsp;





&nbsp;

- 참고 깃허브1 : https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py
- 참고 블로그1 : https://deep-learning-study.tistory.com/563
- 참고 블로그2 : https://debuggercafe.com/transfer-learning-using-efficientnet-pytorch/

&nbsp;

&nbsp;

## EfficientPS 구현

### 논문 리뷰



&nbsp;

### 모델 구현

ㅁㅁ

&nbsp;

&nbsp;