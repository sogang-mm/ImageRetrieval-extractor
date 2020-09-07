import torch.nn as nn

class SiameseNet(nn.Module):
    def __init__(self,backbonequeryNet,backbonerelevantNet):
        super(SiameseNet, self).__init__()
        self.backbonequeryNet = backbonequeryNet
        self.backbonerelevantNet = backbonerelevantNet

    def forward(self, x1, x2, x3):
        output1 = self.backbonequeryNet(x1)
        output2 = self.backbonerelevantNet(x2)
        output3 = self.backbonerelevantNet(x3)
        
        return output1, output2, output3