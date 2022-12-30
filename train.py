# --------------------- basic package ---------------------- #
import numpy as np
import os
import sys
import timeit

from glob import glob
from typing import Dict, List, Tuple, Union

# --------------------- pytorch package ---------------------- #
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

import torchsummary

from tensorboardX import SummaryWriter

# --------------------- file load ---------------------- #
from source.datasets.dataset import *
from source.datasets.dataloader import *

from source.model.backbone import *
from source.model.net import *
from source.model.branch import *

from source.utils.common import *

class Train():
    def __init__(self, params, device, model, optimizer):
        '''
        _summary_ : train code

        Args:
            params (_type_): config parameter
            device (_type_): cpu or gpu
            model (_type_): backbone model class
            optimizer (_type_): optimizer
        '''
        self.params = params
        self.device = device
        self.model = model
        self.optimizer = optimizer

    def run(self):
        print("run")

def collate_fn(batch):
    '''
    _summary_ : 

    Args:
        batch (_type_): _description_
    '''
    pass


def train(cfg_param : os.path, 
          using_gpus : List[int],
          checkpoint : os.path,
          dataset : str) -> None:
    """
    _summary_ : train model

    Args:
        cfg_param (os.path): config parameter file path for model
        using_gpus (List[int]): what you use gpu index. you can write more than one.
        checkpoint (os.path): model checkpoint that you have been learned.
    """
    print("train")

    if cfg_param == None:
        params = makeParam(dataset)
    elif os.path.isfile(cfg_param):
        params = unparseParam(cfg_param)
    else:
        raise ValueError("cfg file path is incorrect!")

    print(f"configs : {params}")

    if len(using_gpus) == 0:
        device = "cpu"
    elif len(using_gpus)  == 1:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = ["cuda:" + str(gpu) for gpu in using_gpus]
    
    print(f"using gpus : {device}")
    

    if checkpoint == None:
        model, optimizer = makeModel()
    elif os.path.isfile(checkpoint):
        model, optimizer = unparseCheckpoint(checkpoint)
    else:
        raise ValueError("checkpoint file path is incorrect!")

    print(f"model info : {torchsummary.summary(model, input_size=(3, 1242, 375), device='cpu')}\
            optimizer : {optimizer}")


    T = Train()
    T.run()


if __name__ == "__main__":
    set_seed()
    args = parse_args()
    train(args.cfg, args.gpus, args.checkpoint, args.dataset)
