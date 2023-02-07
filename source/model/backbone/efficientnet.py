# -*- encoding: etf-8 -*-
# https://github/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py

import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import StochasticDepth


from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from utils import *


class EfficientNet(nn.Module):
    def __init__(self, block_args, num_classes : int = 1000):
        super().__init__()
        self.num_classes = num_classes

        self.has_se = (self.block_args.so_ratio is not None) and (0 < self._block_args.so_ratio <= 1)
        self.id_skip = block_args.id_skip

        if hasattr(nn, 'SiLU'):
            Swish = nn.SiLU
        else:
            # for old pytorch version
            class Swish(nn.Module):
                def forward(self, x):
                    return x * torch.sigmoid(x)


    def forward(self, x):
        return x