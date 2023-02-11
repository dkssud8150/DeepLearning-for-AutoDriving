import torch
import torch.nn as nn
import sys

class CVPRloss(nn.Module):
    def __init__(self, device = torch.device("cpu")):
        super(CVPRloss, self).__init__()
        # mean squared entropy
        self.mseloss = nn.MSELoss().to(device)
        # binary cross entropy
        self.bceloss = nn.BCELoss().to(device)
        # log for binary cross entropy
        self.bcelogloss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device = device)).to(device)
        self.crsEntploss = nn.CrossEntropyLoss().to(device)
    
    def forward(self, output, label):
        loss_value = self.bcelogloss(output, label)

        return loss_value

class KITTIloss(nn.Module):
    def __init__(self, device):
        super(KITTIloss, self).__init__()
        self.loss = nn.CrossEntropyLoss().to(device)
    
    def forward(self, output, label):
        loss_value = self.loss(output, label)

        return loss_value

class BDDloss(nn.Module):
    def __init__(self, device):
        super(BDDloss, self).__init__()
        self.loss = nn.CrossEntropyLoss().to(device)

    def forward(self, output, label):
        loss_value = self.loss(output, label)
        
        return loss_value


def get_criterion(crit = "cvpr", device = torch.device("cpu")):
    if crit == "cvpr":
        return CVPRloss(device=device)
    elif crit == "bdd100k":
        return BDDloss(device=device)
    elif crit == "kitti":
        return KITTIloss(device=device)
    else:
        print("unknown criterion")
        sys.exit(1)



