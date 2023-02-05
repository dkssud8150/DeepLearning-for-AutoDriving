import numpy as np

import torch
import torch.nn as nn

from typing import Any, List, Optional, Type, Union

# https://gaussian37.github.io/dl-concept-covolution_operation/

# 3x3 conv 2개
class BasicBlock(nn.Module):
    expansion : int = 1

    def __init__(self, 
                 in_ch : int, 
                 out_ch : int, 
                 stride : int = 1,
                 dilation : int = 1, # dilation : kernel이 얼마나 간격을 띄우고 연산할지에 대한 값
                 pad : int = 1,
                 groups : int = 1) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.groups = groups
        
        # oh = (( in_h - k_h + 2 * pad ) // stride ) + 1
        # (10,10,3) -> (10 - 3 + 2 * 1) // 1 + 1 = (10,10,3)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, pad=pad, bias=False, dilation=dilation) # 3x3 conv downsample
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, pad=pad, bias=False, dilation=dilation) # 3x3 conv
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.basicblock = nn.Sequential([self.conv1, self.bn1, self.relu,
                                         self.conv2, self.bn2])

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.basicblock(x)

        x += residual
        x = self.relu(x)

        return x

# https://pytorch.org/hub/pytorch_vision_resnet/
# 1x1 -> 3x3 -> 1x1
class Bottleneck(nn.Module):
    def __init__(self,
                 in_ch : int,
                 out_ch : int,
                 stride : int = 1,
                 dilation : int = 1,
                 pad : int = 1,
                 groups : int = 1) -> None:
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.groups = groups

        # width = out_ch * groups
        mid = in_ch / 4

        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=mid, kernel_size=1, stride=1, padding=0, bias=False, dilation=1) # 1x1
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(in_channels=mid, out_channels=mid, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation) # 3x3 downsample
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, out_ch, kernel_size=1, stride=1, padding=0, bias=False, dilation=1) # 1x1
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)


        self.bottleneck = nn.Sequential([self.conv1, self.bn1, self.relu,
                                         self.conv2, self.bn2, self.relu,
                                         self.conv3, self.bn3])

        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.bottleneck(x)

        x += residual
        x = self.relu(x)

        return x


# https://github.com/dkssud8150/dev_yolov3/blob/master/model/yolov3.py
# https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
class ResNet(nn.Module):
    def __init__(self,
                 block : Type[Union[BasicBlock, Bottleneck]],
                 layers : List[int],
                 in_ch : List[int],
                 out_ch : List[int],
                 num_classes : int = 1000,
                 groups : int = 1) -> None:
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.block = block
        self.layers = layers
        self.num_classes = num_classes
        self.groups = groups
        
        self.dilation = 1

        # rgb = 3
        # if input w,h = 224, in_ch is 64
        # (224 - 7 + 2 * 3) // 2 + 1 = 112
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)


        self.layer1 = self.make_layer(block, layers[0], in_ch[0], out_ch[0])
        self.layer2 = self.make_layer(block, layers[1], in_ch[1], out_ch[1])
        self.layer3 = self.make_layer(block, layers[2], in_ch[2], out_ch[2])
        self.layer4 = self.make_layer(block, layers[3], in_ch[3], out_ch[3])


    def make_layer(self,
                    block : Type[Union[BasicBlock, Bottleneck]],
                    layer : int,
                    in_ch : int,
                    out_ch : int,
                    stride : int = 2,
                    dilate : bool = False) -> nn.Sequential:
        if dilate:
            self.dilation *= stride
            stride = 1

        layers = []
        for _ in range(layer):
            layers.append(block(in_ch = in_ch,
                                out_ch = out_ch,
                                stride = stride,
                                dilation = 1, 
                                pad = 1, 
                                groups = 1))

        return nn.Sequential(*layers)


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)




def _resnet(depth, n_classes):
    file = open(f"config/model/resnet{depth}.cfg", "r")
    lines = file.read().split("\n")
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]
    

    for line in lines:
        key, value = line.split("=")
        if key == "block":
            if value == "bottleneck":
                block = Bottleneck
            elif value == "basicblock":
                block = BasicBlock
            
        elif key == "layers":
            layers = list(map(int,value.split(",")))
        elif key == "in_ch":
            in_ch = list(map(int,value.split(",")))
        elif key == "out_ch":
            out_ch = list(map(int,value.split(",")))


    return ResNet(block, layers, in_ch, out_ch, num_classes = n_classes, groups = 1)


def resnet18(n_classes): return _resnet(18, n_classes)
def resnet34(n_classes): return _resnet(34, n_classes)
def resnet50(n_classes): return _resnet(50, n_classes)
def resnet101(n_classes): return _resnet(101, n_classes)