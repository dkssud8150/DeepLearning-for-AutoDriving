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

from source.model.backbone.resnet import *
from source.model.neck import *
from source.model.branch import *

from source.transform.my_transform import *

from source.loss.my_loss import *

from source.utils.common import *


class Train():
    def __init__(self, hyperParams, device, model, optimizer, dataloader, torchwriter):
        '''
        _summary_ : train code

        Args:
            params (_type_): config parameter
            device (_type_): cpu or gpu
            model (_type_): backbone model class
            optimizer (_type_): optimizer
            dataloader (_type_): train data loader
            torchwriter (_type_): device to record loss and accuracy
        '''
        self.hyperParams = hyperParams
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.torchwriter = torchwriter

        self.loss = get_criterion(self.device, crit="cvpr")
        
        self.scheduler_multistep = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[20,40,60], gamma=0.5)

        self.max_epoch = 50

        self.iter = 0

    def run(self):
        total_loss = 0
        self.model.train()
        for epoch in range(self.max_epoch):
            for i, batch in enumerate(self.dataloader):
                if batch is None:
                    continue

                img, label = batch
                # pytorch default type is floatTensor, but if input type is byteTensor(uint8), error would occur.
                img = img.to(self.device, non_blocking=True).float()

                pred = self.model(img)

                loss = self.loss.forward(pred, label)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler_multistep.step(self.iter)
                self.iter += 1

            print("epoch {} / iter {} lr {} loss {}".format(self.epoch, self.iter, get_lr(self.optimizer), loss.item()))
            self.torchwriter.add_scalar("lr", get_lr(self.optimizer), self.iter)
            self.torchwriter.add_scalar("loss", loss, self.iter)
                
            total_loss += loss

            if epoch % 10 == 0:
                checkpoint_path = os.path.join("output", "model_epoch" + str(epoch)+".pth")
                torch.save({"epoch" : epoch,
                            "iteration": self.iter,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.model.state_dict(),
                            "loss": loss},
                            checkpoint_path)

        
def collateFn(batch):
    '''
    _summary_ : 

    Args:
        batch (_type_): 
    '''
    # check exception
    batch = [data for data in batch if data is not None]
    if len(batch) == 0:
        return

    imgs, target_datas = list(zip(*batch))
    imgs = torch.stack([img for img in imgs])

    for i, target in enumerate(target_datas):
        target["batch_idx"] = i
    
    return imgs, target_datas


def training(hyper_param : os.path, 
             using_gpus : List[int],
             checkpoint : os.path,
             dataset : str,
             model_depth : int,
             n_classes : int) -> None:
    """
    _summary_ : train model

    Args:
        cfg_param (os.path): config parameter file path for model
        using_gpus (List[int]): what you use gpu index. you can write more than one.
        checkpoint (os.path): model checkpoint that you have been learned.
    """
    print("train")

    if hyper_param == None:
        params = makeParam(dataset)
    elif os.path.isfile(hyper_param):
        params = unparseParam(hyper_param)
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
    
    #######################################
    # load model layer if model config is exist else make model
    #######################################
    model = unparseModelConfig(model_depth, n_classes)

    #######################################
    # load model and optimizer parameter in checkpoint file if checkpoint is exist
    #######################################
    if model_depth != None:
        model, optimizer = unparseCheckpoint(checkpoint, model, params)
    else:
        raise ValueError("checkpoint or config file path is incorrect!")

    # input size = (c,w,h)
    print(f"model info : {torchsummary.summary(model, input_size=(3, 224, 224), batch_size=2, device='cpu')} optimizer : {optimizer}")


    ####################################
    # Transform
    ####################################
    train_transform = getTransform(hyper_param = params, is_train=True)


    ####################################
    # dataloader 
    ####################################
    dataset = datasets(dataset_name='cvpr',transform=train_transform, is_train=True)
    dataloader = DataLoader(dataset, batch_size=params["train_bs"], num_workers=0, pin_memory=True, drop_last=True, shuffle=True, collate_fn=collateFn)

    # should check the dataloader data using next.
    # img, batch = next(iter(dataloader))

    ####################################
    # Train
    ####################################
    model.train()
    model.initialize_weights()
    model.to(device)

    # tensorboard
    torchwriter = SummaryWriter("output/tensorboard")


    T = Train(hyper_param, device, model, optimizer, dataloader, torchwriter)
    T.run()


if __name__ == "__main__":
    set_seed()
    args = parse_args()
    training(args.param, args.gpus, args.checkpoint, args.dataset, args.model_depth, args.n_class)
