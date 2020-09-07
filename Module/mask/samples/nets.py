import torch.nn.functional as F
import torch.nn as nn

import torchvision



class backbonequeryNet(nn.Module):
    def __init__(self):
        # query
        super(backbonequeryNet ,self).__init__()
        self.backbone_query = torchvision.models.resnet50(pretrained=False)
        state_dict_query = self.backbone_query.state_dict()

        num_features_query = self.backbone_query.fc.in_features

        self.backbone_query = nn.Sequential(*list(self.backbone_query.children())[:-2])
        model_dict_query = self.backbone_query.state_dict()
        model_dict_query.update({k: v for k, v in state_dict_query.items() if k in model_dict_query})
        self.backbone_query.load_state_dict(model_dict_query)

        # query
        self.maxpooling_query = nn.MaxPool2d(7, stride=1)
        self.fc_query = nn.Linear(num_features_query, 1000)

    def forward(self, x):
        x = self.backbone_query(x)
        maxpool_query = self.maxpooling_query(x)
        out_temp_query = self.fc_query(maxpool_query.view(maxpool_query.size(0), -1))
        out_query = F.normalize(out_temp_query, p=2, dim=1)  # L2 normalized

        return out_query


class backbonerelevantNet(nn.Module):
    def __init__(self):
        # non-relevant, relevant
        super(backbonerelevantNet, self).__init__()
        self.backbone = torchvision.models.resnet50(pretrained=False)
        state_dict = self.backbone.state_dict()

        num_features = self.backbone.fc.in_features

        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        model_dict = self.backbone.state_dict()
        model_dict.update({k: v for k, v in state_dict.items() if k in model_dict})
        self.backbone.load_state_dict(model_dict)

        # non-relevant, relevant
        self.maxpooling = nn.MaxPool2d(7, stride=1)
        self.fc = nn.Linear(num_features, 1000)

    def forward(self, x):
        x = self.backbone(x)
        maxpool = self.maxpooling(x)
        out_temp = self.fc(maxpool.view(maxpool.size(0), -1))
        out = F.normalize(out_temp, p=2, dim=1)  # L2 normalized

        return out


class SiameseNet(nn.Module):
    def __init__(self, backbonequeryNet, backbonerelevantNet):
        super(SiameseNet, self).__init__()
        self.backbonequeryNet = backbonequeryNet
        self.backbonerelevantNet = backbonerelevantNet

    def forward(self, x1, x2, x3):
        output1 = self.backbonequeryNet(x1)
        output2 = self.backbonerelevantNet(x2)
        output3 = self.backbonerelevantNet(x3)

        return output1, output2, output3