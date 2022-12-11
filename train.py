import os, sys, timeit
import argparse

from glob import glob

import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

from tensorboardX import SummaryWriter

from typing import Dict, List, Tuple

def parse_args():
    parser = argparse.ArgumentParser(description="arguments")
    parser.add_argument("--gpus", type=int, nargs="+",
                        help="List of GPU device id", default=[])
    # parser.add_argument("--mode", type=str,
    #                     help="train / val / test", default="train")
    parser.add_argument("--cfg", type=str,
                        help="model config path", default=None)
    parser.add_argument("--checkpoint", type=str,
                        help="model checkpoint path", default=None)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    args = parser.parse_args()
    return args

def collate_fn(batch):
    pass



def train(cfg_param : os.path = None, 
          using_gpus : List[int] = None) -> None:
    print("train")



if __name__ == "__main__":
    args = parse_args()    
    train()
