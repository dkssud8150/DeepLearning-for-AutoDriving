import cv2
import numpy as np

import torch
import torchvision.transforms as transforms

import albumentations as A
from albumentations.pytorch import transforms

def getTransform(hyper_param = None, is_train=True):
    # https://dacon.io/codeshare/4379
    # https://89douner.tistory.com/312
    # must search for the most famous tranform method.
    if is_train:
        data_transform = A.Compose([
                                    A.RandomCrop(width=256, height=256),
                                    A.HorizontalFlip(p=0.5),
                                    A.RandomBrightnessContrast(p=0.2),
                                    transforms.ToTensorV2(transpose_mask=True),
                                    ])
        # data_transform = transforms.Compose([ResizeImg(new_size=(int(hyper_param["img_size"][0]), int(hyper_param["img_size"][1]))),
        #                                      ToTensor()])
    else:
        data_transform = transforms.Compose([
                                             ResizeImg(new_size=(hyper_param["img_size"][0], hyper_param["img_size"][1])),
                                             ToTensor()])
    
    return data_transform



class ResizeImg(object):
    def __init__(self, new_size, interpolation = cv2.INTER_LINEAR):
        self.new_size = tuple(new_size)
        self.interpolation = interpolation

    def __call__(self, data):
        img, target_data = data
        img = img.resize(self.new_size, self.interpolation)
        target_data["mask_core"] = cv2.resize(target_data["mask_core"], self.new_size, self.interpolation)
        target_data["mask_edge"] = cv2.resize(target_data["mask_edge"], self.new_size, self.interpolation)
        return img, target_data
        # bbox의 경우 label은 normalize하면, resize를 다시 해주지 않아도 된다. 나중에 width, height를 resize된 사이즈로 곱하게 되면 resize된 label로 만들어진다. 그러나 segmentation의 경우 동일한 이미지 크기의 영역 표시로 되어 있는 mask가 label이므로 동일한 방식으로 resize

class ToTensor(object):
    def __init__(self,):
        pass
    def __call__(self, data):
        img, label = data
        img = torch.tensor(np.transpose(np.array(img, dtype=float) / 255, (2,0,1)), dtype=torch.float32)
        label = torch.FloatTensor(np.array(label))

        return img, label