# --------------------- basic package ---------------------- #
import os
import sys
import numpy as np
from glob import glob

from typing import Dict, Union, Tuple, List
# --------------------- pytorch package ---------------------- #
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import torchvision

# --------------------- image processing package ---------------------- #
from PIL import Image

"""
문제 1. color 이미지보다 label 이미지가 약 8배 많음... 원래는 같음. color 데이터가 덜 풀린듯.
문제 2. 8배가 많은 이유는 ConvertVideotoCSV.py 에 있는 듯. 이를 분석해야 함. => 제출을 위한 CSV를 만드는데, 필요한 오차들을 계산하기 위한 prediction value들을 추가해주는 용도인듯
문제 3. train_video_list 폴더에 있는 파일들 중 한 파일마다 한 timestamp 마다의 <image - label> 로 구성되어 있음.
문제 4. label format
    The training images labels are encoded in a format mixing spatial and label/instance information:
    - All the images are the same size (width, height) of the original images
    - Pixel values indicate both the label and the instance.
    - Each label could contain multiple object instances.
    - int(PixelValue / 1000) is the label (class of object)
    - PixelValue % 1000 is the instance id
    - For example, a pixel value of 33000 means it belongs to label 33 (a car), is instance #0, while the pixel value of 33001 means it also belongs to class 33 (a car) , and is instance #1. These represent two different cars in an image.
문제 5. opencv가 PIL보다 속도는 빠르나, torchvision과의 호환성이 PIL이 더 좋다. 그래서 딥러닝을 할 시에는 PIL을, 딥러닝 이외의 영상처리에서는 OpenCV를 사용하고자 함.
"""

class datasets(Dataset):
    
    base_dir = "C:/Users/dkssu/github/dl4ad/data/"

    def __init__(self,
                 params : Dict,
                 dataset_name : str,
                 transform = None, 
                 is_train : bool = True):
        super(datasets, self).__init__()

        self.params = params
        self.transform = transform
        self.is_train = is_train
        self.base_dir += dataset_name
        self.dataset_name = dataset_name                    

        if self.is_train:
            txt_dir = self.base_dir + "/train_video_list/"
            txt_files = sorted(glob(txt_dir + "/*.txt"), key = lambda x : x.split("/")[-1])

            self.img_data = []
            self.label_data = []

            for txt_file in txt_files:
                with open(txt_file, 'r', encoding="utf-8") as t:
                    lines = t.readlines()
                    for line in lines:
                        img, label = line.split("\t")
                        img_path = img.split("\\")[-1]
                        label_path = label.split("\\")[-1].strip()

                        self.img_data.append(self.base_dir + "/train_color/" + img_path)
                        self.label_data.append(self.base_dir + "/train_label/" + label_path)
        else:
            print('not exist valid data, so i split train data into 8:2 randomly or stratified method, so before the validation test, i watch and anaysis num of labeled class in train data')
            self.img_dir = self.base_dir + "/valid_set/"
            self.img_data = sorted(glob(self.img_dir + "/*.jpg"), key = lambda x : x.split("/")[-1])

        
        print(f"data length : {len(self.img_data)}")


    def __getitem__(self, idx):
        img_file = self.img_data[idx]
        
        img = Image.open(img_file).convert(mode="RGB")
        img_w, img_h = img.shape[:2]

        # train
        if self.is_train:
            anno_file = self.label_data[idx]
            if not os.path.isdir(anno_file):
                return None

            anno = Image.open(anno_file).convert(mode="RGB")

            labels = []
            instance_ids = []
            if self.dataset_name == "bdd100k":
                print("bdd100k")
            elif self.dataset_name == "nuscenes":
                print("nucenes")
            elif self.dataset_name == "cvpr":
                print("cvpr 2018 dataset")
                anno = np.asarray(anno) # array는 copy=True, asarray는 copy=False
                
                label = torch.as_tensor((anno / 1000), dtype=torch.int64)       # semantic
                instance_id = torch.as_tensor((anno % 1000), dtype=torch.int64) # instance
                
            labels = torch.as_tensor((np.unique(label)[1:], ), dtype=torch.int64)
            obj_ids = np.unique(instance_id)[1:] # remove background
            num_obj = len(obj_ids)

            mask = anno == obj_ids[:, None, None]
            mask = torch.as_tensor(mask, dtype=torch.uint8)

            image_id = torch.tensor([idx])

            is_crowd = torch.zeros((num_obj, ), dtype=torch.int64)
            batch_idx = torch.zeros((num_obj, ), dtype=torch.int64) # 객체 개수만큼 생성

            target_data = {}
            target_data["label"] = labels
            target_data["mask"] = mask
            target_data["image_id"] = image_id
            target_data["is_crowd"] = is_crowd
            target_data["batch_idx"] = batch_idx

            if self.transform is not None:
                img, label = self.transform((img, target_data))

            # TODO: to be continue..
            '''
            https://github.com/dkssud8150/dev_yolov3/blob/master/dataloader/yolo_data.py
            https://dacon.io/en/codeshare/4379
            https://colab.research.google.com/drive/1CrDusRQdmRELrWyBdt4uS3AoAr8izrWI?authuser=1&hl=ko
            '''

            return img, target_data

        # valid
        else:
            target_data = {}
            if self.transform is not None:
                img, _ = self.transform((img, target_data))
            
            return img, None

    def __len__(self):
        return len(self.img_data)


