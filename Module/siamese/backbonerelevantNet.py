import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import torch

class backbonerelevantNet(nn.Module):
    def __init__(self):
        #non-relevant, relevant
        super(backbonerelevantNet,self).__init__()
        self.backbone = torchvision.models.resnet50(pretrained=False)
        state_dict = self.backbone.state_dict()
        
        num_features = self.backbone.fc.in_features
        
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        model_dict = self.backbone.state_dict()
        model_dict.update({k: v for k, v in state_dict.items() if k in model_dict})
        self.backbone.load_state_dict(model_dict)       
       
        #non-relevant, relevant
        self.maxpooling = nn.MaxPool2d(7, stride=1)
        self.fc = nn.Linear(num_features, 1000)
        
    def forward(self,x):
        x = self.backbone(x)
        maxpool = self.maxpooling(x)
        out_temp = self.fc(maxpool.view(maxpool.size(0), -1))
        out = F.normalize(out_temp, p=2, dim=1) #L2 normalized
            
        return out 
