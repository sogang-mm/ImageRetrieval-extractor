import torch.nn as nn
import numpy as np
import torchvision
import torch
import torch.nn.functional as F


class backbonequeryNet(nn.Module):
    def __init__(self):
        #query
        super(backbonequeryNet,self).__init__()
        self.backbone_query = torchvision.models.resnet50(pretrained=False)
        state_dict_query = self.backbone_query.state_dict()
        
        num_features_query = self.backbone_query.fc.in_features
        
        self.backbone_query = nn.Sequential(*list(self.backbone_query.children())[:-2])
        model_dict_query = self.backbone_query.state_dict()
        model_dict_query.update({k: v for k, v in state_dict_query.items() if k in model_dict_query})
        self.backbone_query.load_state_dict(model_dict_query)
        
        #query
        self.maxpooling_query = nn.MaxPool2d(7, stride=1)
        self.fc_query = nn.Linear(num_features_query, 1000)    
        
    def forward(self, x):
        x = self.backbone_query(x)
        maxpool_query = self.maxpooling_query(x)
        out_temp_query = self.fc_query(maxpool_query.view(maxpool_query.size(0), -1))
        out_query = F.normalize(out_temp_query, p=2, dim=1) #L2 normalized
            
        return out_query