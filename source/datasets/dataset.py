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
import cv2
from PIL import Image

"""
문제 1. ConvertVideotoCSV.py => 제출을 위한 CSV를 만드는데, 필요한 오차들을 계산하기 위한 prediction value들을 추가해주는 용도인듯
문제 2. train_video_list 폴더에 있는 파일들 중 한 파일마다 한 timestamp 마다의 <image - label> 로 구성되어 있음.
문제 3. opencv가 PIL보다 속도는 빠르나, torchvision과의 호환성이 PIL이 더 좋다. 그래서 딥러닝을 할 시에는 PIL을, 딥러닝 이외의 영상처리에서는 OpenCV를 사용하고자 함.
"""

class datasets(Dataset):
    
    base_dir = "data/"
    total_label = {33:'car', 34:"motorbicycle", 35:"bicycle", 36:"person", 38:"truck", 39:"bus", 40:"tricycle"}
    classes = list(total_label.keys())

    def __init__(self,
                 dataset_name : str,
                 transform = None, 
                 is_train : bool = True):
        super(datasets, self).__init__()

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
        
        if not os.path.isfile(img_file):
             return None
        
        img = Image.open(img_file).convert(mode="RGB")
        img_w, img_h = img.size

        # train
        if self.is_train:
            anno_file = self.label_data[idx]
            if not os.path.isfile(anno_file):
                return None

            anno = cv2.imread(anno_file)

            if self.dataset_name == "bdd100k":
                print("bdd100k")
            elif self.dataset_name == "nuscenes":
                print("nucenes")
            elif self.dataset_name == "cvpr":
                print("cvpr 2018 dataset")
                
                # Image에서 바로 tensor로는 변경 불가능. Image -> numpy -> tensor
                anno = np.asarray(anno) # array는 copy=True, asarray는 copy=False

                # https://www.kaggle.com/code/ishootlaser/cvrp-2018-starter-kernel-u-net-with-resnet50
                # https://www.kaggle.com/code/kmader/data-preprocessing-and-unet-segmentation-gpu
                # semantic id # TODO
                label = (anno * ((anno >= self.classes[0] * 1000) & (anno < (self.classes[-1]+1) * 1000))).astype(np.uint16)
                # instance id # TODO
                instance_id = torch.tensor((anno % 1000), dtype=torch.int64)
                
            labels = torch.as_tensor((np.unique(label)[1:], ), dtype=torch.int64)
            obj_ids = np.unique(instance_id)[1:] # remove background
            num_obj = len(obj_ids)

            mask = anno == obj_ids[:, None, None]
            mask = torch.as_tensor(mask, dtype=torch.uint8)

            image_id = torch.tensor([idx])

            is_crowd = torch.zeros((num_obj, ), dtype=torch.int64) # torch.size([num_obj]) : tensor([0., 0., 0., ...])
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


