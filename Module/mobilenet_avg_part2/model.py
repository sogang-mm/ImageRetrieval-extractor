import torch.nn as nn
from torchvision import models
from collections import OrderedDict
import torch


class L2N(nn.Module):

    def __init__(self, eps=1e-6):
        super(L2N, self).__init__()
        self.eps = eps

    def forward(self, x):
        return x / (torch.norm(x, p=2, dim=1, keepdim=True) + self.eps).expand_as(x)

    def __repr__(self):
        return f'{self.__class__.__name__}(eps={self.eps})'


class MobileNet_AVG_PART2(nn.Module):
    def __init__(self):
        super(MobileNet_AVG_PART2, self).__init__()
        self.base = nn.Sequential(
            OrderedDict([*list(models.mobilenet_v2(pretrained=True).features.named_children())]))
        # [:2], [:4], [:14], [:]
        self.child_base = nn.Sequential(OrderedDict([*list(self.base.named_children())[:4]]))
        self.child_base.add_module('relu6', nn.ReLU6(inplace=True))
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.norm = L2N()

    def forward(self, x):
        x = self.child_base(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.norm(x)
        return x


def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        print(key_item_1[0], key_item_2[0])
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')
