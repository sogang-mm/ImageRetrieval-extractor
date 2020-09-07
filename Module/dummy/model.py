import torch.nn as nn
from torchvision import models


class Dummy(nn.Module):
    def __init__(self):
        super(Dummy, self).__init__()
        self.base = models.resnet18(pretrained=True)

    def forward(self, x):
        x = self.base(x)
        return x
