import cv2
import numpy as np

import torch
import torchvision.transforms as transforms

def getTransform(hyper_param = None, is_train=True):
    if is_train:
        data_transform = transforms.Compose([ResizeImg(new_size=(hyper_param["img_width"], hyper_param["img_height"])),
                                             ToTensor()])
    else:
        data_transform = transforms.Compose([ResizeImg(new_size=(hyper_param["img_width"], hyper_param["img_height"])),
                                             ToTensor()])
    
    return data_transform



class ResizeImg(object):
    def __init__(self, new_size, interpolation = cv2.INTER_LINEAR):
        self.new_size = tuple(new_size)
        self.interpolation = interpolation

    def __call__(self, data):
        img, label = data
        img = cv2.resize(img, self.new_size, interpolation=self.interpolation)
        label = cv2.resize(label, self.new_size, interpolation=self.interpolation)
        return img, label
        # bbox의 경우 label은 normalize하면, resize를 다시 해주지 않아도 된다. 나중에 width, height를 resize된 사이즈로 곱하게 되면 resize된 label로 만들어진다. 그러나 segmentation의 경우 동일한 이미지 크기의 영역 표시로 되어 있는 mask가 label이므로 동일한 방식으로 resize

class ToTensor(object):
    def __init__(self,):
        pass
    def __call__(self, data):
        img, label = data
        img = torch.tensor(np.transpose(np.array(img, dtype=float) / 255, (2,0,1)), dtype=torch.float32)
        label = torch.FloatTensor(np.array(label))

        return img, label