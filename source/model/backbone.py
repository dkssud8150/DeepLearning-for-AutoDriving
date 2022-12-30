import numpy as np

import torch
import torch.nn as nn

from typing import Any, List, Optional, Type, Union

class BasicBlock(nn.Module):
    expansion : int = 1

    def __init__(self, 
                 in_ch : int, 
                 out_ch : int, 
                 stride : int = 1,
                 dilation : int = 1, # dilation : kernel이 얼마나 간격을 띄우고 연산할지에 대한 값
                 pad : int = 1,
                 ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride
        self.dilation = dilation
        self.pad = pad

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, pad=pad, bias=False, dilation=dilation) # 3x3 conv
        self.bn1 = nn.BatchNorm2d
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, pad=pad, bias=False, dilation=dilation) # 3x3 conv
        self.bn2 = nn.BatchNorm2d

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        x += identity
        x = self.relu(x)

        return x


class Bottleneck(nn.Module):
    expansion : int = 4

    def __init__(self,
                 in_ch : int,
                 out_ch : int,
                 stride : int = 1,
                 dilation : int = 1,
                 pad : int = 1) -> None:
        super().__init__()

        # TODO: to be continue..
        ## https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
        ## https://dkssud8150.github.io/posts/cnn/

