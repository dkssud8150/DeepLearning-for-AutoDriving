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

        self.basicblock = nn.Sequential(*[self.conv1, self.bn1, self.relu,
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
    expansion = 4

    def __init__(self,
                 in_ch : int,
                 out_ch : int,
                 stride : int = 1,
                 dilation : int = 1,
                 pad : int = 1,
                 groups : int = 1,
                 downsample = None) -> None:
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.groups = groups
        self.downsample = downsample

        width = out_ch * groups

        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=width, kernel_size=1, padding=0, bias=False, dilation=1) # [B,in_c,in_h,in_w]
        self.bn1 = nn.BatchNorm2d(width)  # [B,in_c,in_h,in_w]
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation) # [B,in_c,in_h//2,in_w//2] # 3x3 downsample
        self.bn2 = nn.BatchNorm2d(width) # [B,in_c,in_h,in_w]
        self.conv3 = nn.Conv2d(width, out_ch * self.expansion, kernel_size=1, padding=0, bias=False, dilation=1) # 1x1 # [B,in_c*4,in_h,in_w]
        self.bn3 = nn.BatchNorm2d(out_ch * self.expansion) # [B,in_c,in_h,in_w]
        self.relu = nn.ReLU(inplace=True) # [B,in_c,in_h,in_w]

        # self.bottleneck = nn.Sequential(*[self.conv1, self.bn1, self.relu,
        #                                   self.conv2, self.bn2, self.relu,
        #                                   self.conv3, self.bn3])

        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # torch.size([2,64,56,56]) [2,256,56,56] * 2
        # torch.size([2,256,56,56]) [2,512,28,28] * 3
        residual = x
        x = self.conv1(x)
        # torch.size([2,64,56,56]) [2,64,56,56] * 2
        # torch.size([2,128,56,56]) [2,128,28,28] * 3
        x = self.bn1(x) 
        # torch.size([2,64,56,56]) [2,64,56,56] * 2
        # torch.size([2,128,56,56]) [2,128,28,28] * 3      
        x = self.relu(x)
        # torch.size([2,64,56,56]) [2,64,56,56] * 2
        # torch.size([2,128,56,56]) [2,128,28,28] * 3

        x = self.conv2(x)
        # torch.size([2,64,56,56]) [2,64,56,56] * 2
        # torch.size([2,128,28,28]) [2,128,28,28] * 3
        x = self.bn2(x)
        # torch.size([2,64,56,56]) [2,64,56,56] * 2
        # torch.size([2,128,28,28]) [2,128,28,28] * 3
        x = self.relu(x)
        # torch.size([2,64,56,56]) [2,64,56,56] * 2
        # torch.size([2,128,28,28]) [2,128,28,28] * 3

        x = self.conv3(x)
        # torch.size([2,256,56,56]) [2,256,56,56] * 2
        # torch.size([2,512,28,28]) [2,512,28,28] * 3
        x = self.bn3(x)
        # torch.size([2,256,56,56]) [2,256,56,56] * 2
        # torch.size([2,64,28,28]) [2,512,28,28] * 3

        if self.downsample is not None:
            # [2,64,56,56]
            # [2,256,28,28]
            residual = self.downsample(residual)
            # [2,256,56,56]
            # [2,512,28,28]

        # torch.size([2,256,56,56]) [2,256,56,56] * 2
        # torch.size([2,512,28,28]) [2,512,56,56] * 3
        x += residual
        # torch.size([2,256,56,56]) [2,256,56,56] * 2
        # torch.size([2,64,56,56]) [2,512,56,56] * 3
        x = self.relu(x)
        # torch.size([2,256,56,56]) [2,256,56,56] * 2
        # torch.size([2,64,56,56]) [2,512,56,56] * 3

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

        self.in_ch = in_ch[0]
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
        self.fc = nn.Linear(512 * block.expansion, num_classes)


        self.layer1 = self.make_layer(block, layers[0], in_ch[0], stride=1)
        self.layer2 = self.make_layer(block, layers[1], in_ch[1], stride=2)
        self.layer3 = self.make_layer(block, layers[2], in_ch[2], stride=2)
        self.layer4 = self.make_layer(block, layers[3], in_ch[3], stride=2)


    def make_layer(self,
                    block : Type[Union[BasicBlock, Bottleneck]],
                    layer : int,
                    in_ch : int,
                    stride : int = 1) -> nn.Sequential:

        layers = []

        if stride != 1 or self.in_ch != in_ch * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.in_ch, in_ch * block.expansion, kernel_size=1, stride=stride, padding=0, bias=False, dilation=1), 
                        nn.BatchNorm2d(in_ch * block.expansion))

        layers.append(block(self.in_ch, in_ch, stride=stride, downsample=downsample))
        self.in_ch = in_ch * block.expansion

        for _ in range(1, layer):
            layers.append(block(in_ch = self.in_ch,
                                out_ch = in_ch,
                                dilation = 1, 
                                pad = 1, 
                                groups = 1))

        return nn.Sequential(*layers)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # torch.size([2,3,224,224]) B,C,H,W
        x = self.conv1(x) # 7x7, stride=2, out_c=64
        # torch.size([2,64,112,112]) [B,64,in_h//2,in_w//2]
        x = self.bn1(x)
        # torch.size([2,64,112,112]) [B,in_c,in_h,in_w]
        x = self.relu(x)
        # torch.size([2,64,112,112]) [B,in_c,in_h,in_w]
        x = self.maxpool(x) # 2x2 pool
        # torch.size([2,64,56,56]) [B,in_c,in_h//2,in_w//2]
        
        x = self.layer1(x)
        # torch.size([2,256,56,56]) [B,in_c*4,in_h,in_w]
        x = self.layer2(x)
        # torch.size([2,512,28,28]) [B,in_c*4,in_h,in_w]
        x = self.layer3(x)
        # torch.size([2,1024,14,14]) [B,in_c*4,in_h,in_w]
        x = self.layer4(x)
        # torch.size([2,2048,7,7]) [B,in_c*4,in_h,in_w]

        x = self.avgpool(x) # 2x2 pool
        # torch.size([2, 2048, 1, 1])
        x = torch.flatten(x, 1)
        # torch.size([2, 2048])
        x = self.fc(x)
        # torch.size([2, 10])

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
            layers = list(map(int, value.split(",")))
        elif key == "in_ch":
            in_ch = list(map(int, value.split(",")))
        elif key == "out_ch":
            out_ch = list(map(int, value.split(",")))


    return ResNet(block, layers, in_ch, out_ch, num_classes = n_classes, groups = 1)


def resnet18(n_classes): return _resnet(18, n_classes)
def resnet34(n_classes): return _resnet(34, n_classes)
def resnet50(n_classes): return _resnet(50, n_classes)
def resnet101(n_classes): return _resnet(101, n_classes)