# --------------------- basic package ---------------------- #
import numpy as np
import os
import sys
from glob import glob
from typing import Dict, Union, Tuple, List
# --------------------- pytorch package ---------------------- #
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import torchvision

# --------------------- EDA package ---------------------- #
from PIL import Image
import cv2

class datasets(Dataset):
    
    base_dir = "C:/Users/dkssu/github/dl4ad/data/"

    def __init__(self,
                 params : Dict,
                 dataset_name : str,
                 transform = None, 
                 is_train : bool = True):
        super(datasets, self)._init__()

        self.params = params
        self.transform = transform
        self.is_train = is_train
        self.base_dir += dataset_name

        if self.is_train:
            self.img_dir = self.base_dir + "/train/PNGimages"
            self.anno_dir = self.base_dir + "/train/Annotations/semantic/"
            self.anno_data = sorted(glob(self.anno_dir + "/*"), key = lambda x : x.split("/")[-1])
        else:
            self.img_dir = self.base_dir + "/valid/PNGimages"
            self.anno_dir = self.base_dir + "/valid/Annotations/semantic/"

        self.img_data = sorted(glob(self.img_dir + "/*"), key = lambda x : x.split("/")[-1])
        print(f"data length : {len(self.img_data)}")


    def __getitem__(self, idx):
        img_file = self.img_data[idx]
        
        img_np = np.fromfile(img_file, dtype=np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        img_w, img_h = img.shape[:2]

        # train
        if self.is_train:
            if os.path.isdir(self.anno_dir):
                anno_file = self.anno_data[idx]
                anno_np = np.fromfile(anno_file, dtype = np.uint8)
                anno = cv2.imdecode(anno_np, cv2.IMREAD_COLOR)

                # TODO: to be continue..
                '''
                https://github.com/dkssud8150/dev_yolov3/blob/master/dataloader/yolo_data.py
                https://dacon.io/en/codeshare/4379
                https://jins-sw.tistory.com/39
                https://colab.research.google.com/drive/1CrDusRQdmRELrWyBdt4uS3AoAr8izrWI?authuser=1&hl=ko
                https://colab.research.google.com/drive/1dWg0nx7KEYGSH05heY2_z5hosHBK3EbP?authuser=1&hl=ko
                '''

                


                



            else:
                raise ValueError(f"{self.anno_dir} is not exist")
        # valid
        else:
            pass



