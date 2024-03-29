import argparse
import numpy as np
import os
import random
import sys

from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision

# --------------------- file load ---------------------- #
from ..model.backbone.resnet import *

def set_seed(seed : int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def generate_data(input_dim, num_samples, num_batches):
    x = torch.rand((num_samples, input_dim)).reshape(-1, num_batches, input_dim)
    y = torch.randint(2, (num_samples,)).reshape(-1, num_batches).float()

    return x, y


def parse_args():
    parser = argparse.ArgumentParser(description="arguments")
    parser.add_argument("-g","--gpus", type=int, nargs="+",
                        help="List of GPU device id", default=[])
    parser.add_argument("-p","--param", type=str, default="config/kitti_resnet.json",
                        help="hyper parameter files")
    parser.add_argument("--depth", type=int,
                        help="model layer number", default=50)
    parser.add_argument("--checkpoint", type=str,
                        help="model checkpoint path", default=None)
    parser.add_argument("-d","--dataset", type=str,
                        help="what you use dataset", default="kitti")
    parser.add_argument("--mode", type=str,
                        help="train / valid / test", default="train")
    parser.add_argument("--n_class", type=int, default=10,
                        help="what number you want to classify")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    args = parser.parse_args()
    return args


def makeParam(dataset : str) -> Dict:
    '''
    _summary_ : if config parameter file is not exist, make it as default.

    Returns:
        _type_: config Dictionary for model
    '''
    print(f"🚫 you dont put your config file, so i should make config")
    if dataset == "kitti":
        params = {
            "seed" : 42,
            "img_size" : (1242, 375),
            "train_bs" : 16,
            "valid_bs" : 16,
            "num_classes" : 25,
            "n_fold" : 5,
            "model_name" : "regnet",
            "optimizer" : "AdamW", # adam, SGD, adamw
            "lr" : 1e-4,
            "scheduler" : "CosineAnnealingLR",
            "warmup_epochs" : 0,
            "weight_decay" : 1e-6,
            "sgd_momentum" : 0.999,
        }
    else:
        raise ValueError("dataset is incorrect!")

    return params

def unparseParam(cfg_param : os.path) -> Dict:
    """
    _summary_ : convert from config parameter file path to Dictionary

    Args:
        cfg_param (os.path): config parameter file path for model
    
    _refer_ : https://bluese05.tistory.com/31
    """
    import importlib
    
    params = importlib.import_module(cfg_param.split(".")[0].replace("/",".")).cfg
    return params

def unparseModelConfig(depth : int, n_classes : int):
    if depth == None:
        model = torchvision.models.resnet50(pretrained=True)
    if depth == 18:
        model = resnet18(n_classes)
    if depth == 34:
        model = resnet34(n_classes)
    if depth == 50:
        model = resnet50(n_classes) # RegNet()
    if depth == 101:
        model = resnet101(n_classes)

    return model

def unparseCheckpoint(checkpoint : Union[str, None], model, params) -> nn.Module:
    optimizer = optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

    if checkpoint != None:
        checkpoint = torch.load(checkpoint)

        print(f"\nyour checkpoint : {checkpoint}\n")

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer

def makeModel() -> nn.Module:
    print("\n🚫 you dont put your checkpoint, so you should make and download model")
    model = torchvision.models.resnet34(pretrained=True, progress=True)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.001)
    return model, optimizer



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]