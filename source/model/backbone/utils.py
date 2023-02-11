import torch
import torch.nn as nn
from torch.nn import functional as F

import math
import collections
from functools import partial

##################################################################################################################
# GlobalParams / BlockArgs
# Swish / MemoryEfficientSwish : Two Implementations of the method
# round_filters / round_repeats : Functions to calculate params for scaling model width and depth
# drop_connect : structural design
# get_same_padding_conv2d : 
    # Conv2dDynamicSamePadding
    # Conv2dStaticSamePadding
# get_same_padding_maxpool2d : 
    # MaxPool2dDynamicSamePadding
    # MaxPool2dDynamicSamePadding
    # it's an additional function, not used in EfficientNet, but can be used in other model(like EfficientDet).
##################################################################################################################

MODELS = ["EfficientNet",
          "EfficientNet_b0",
          "EfficientNet_b1",
          "EfficientNet_b2",
          "EfficientNet_b3",
          "EfficientNet_b4",
          "EfficientNet_b5",
          "EfficientNet_b6",
          "EfficientNet_b7",
          "EfficientNet_v2_s",
          "EfficientNet_v2_m",
          "EfficientNet_v2_l",
          ]


BlockArgs = collections.namedtuple('BlockArgs', ['num_repeat', 'kernel_size', 'stride', 'expand_ratio', 'input_filters', 'output_filters', 'so_ratio', 'id_skip'], rename=False) # https://zzsza.github.io/development/2020/07/05/python-namedtuple/

#########################################################
# A memory-efficient implementation of Swish function
#########################################################
class SwishImplementation(torch.autograd.Function): # TODO: swish가 무엇인지
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Memory_EfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)
          


#########################################################
# define dynamic and static conv2d-padding layer
#########################################################
def get_same_padding_conv2d(img_size=None):
    """
    if you want static padding for specified image size, otherwise dynamic padding
    but, static padding is necessary for ONNX exporting of models
    when stride == 1, can the output size be the same as input size. otherwise, output size is ceil(input size/stride)

    Returns:
        Conv2dDynamicSamePadding or Conv2dStaticSamePadding
    """

    if img_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, img_size=img_size)
    

class Conv2dDynamicSamePadding(nn.Conv2d):
    def __init__(self, in_c, out_c, kernel_size, stride = 1,dilation = 1,groups = 1, bias = True):
        super().__init__(in_c, out_c, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2 # make list of two value
    
    def forward(self, x):
        ih, iw = x.size()[-2:] # x.shape : B,C,H,W
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        # out_w * stride + kernel * dilation + 1 - input_w
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation + 1 - iw, 0)

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Conv2dStaticSamePadding(nn.Conv2d):
    def __init__(self, in_c, out_c, kernel_size, stride=1, img_size=None, **kwargs):
        super().__init__(in_c, out_c, kernel_size, stride, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        assert img_size is not None
        ih, iw = (img_size, img_size) if isinstance(img_size, int) else img_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)

        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = nn.Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


#########################################################
# define dynamic and static maxpool2d-padding layer
#########################################################
def get_same_padding_maxpool2d(img_size=None):
    if img_size is None:
        return MaxPool2dDynamicSamePadding
    else:
        return partial(MaxPool2dStaticSamePadding, img_size=img_size) ## TODO: partial의 용도


class MaxPool2dDynamicSamePadding(nn.MaxPool2d):
    def __init__(self):
        pass

    def forward(self, x):
        pass


class MaxPool2dStaticSamePadding(nn.MaxPool2d):
    def __init__(self):
        pass

    def forward(self, x):
        pass


def drop_connect(inputs, p, is_train=True): # TODO: drop connect의 정의 및 효과
    """
    :param inputs (tensor : BCWH) = input of this structure
    :param p (float : 0.0~1.0) = Probablilty of drop connection
    :param is_train (bool) = train / valid / test
    
    return:
        output = output after drop connection
    """

    assert 0 <= p <= 1, 'p must be in range of [0,1]'
    
    if not is_train:
        return inputs

    batch_size = inputs.shape[0]
    keep_prob = 1 - p

    random_tensor = keep_prob + torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)

    output = inputs / keep_prob * binary_tensor
    return output


class MBConv(nn.Module):
    def __init__(self, ):
        pass