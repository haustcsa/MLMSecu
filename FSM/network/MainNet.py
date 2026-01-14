
import torch
import torch.nn as nn

import FSMC_ResNet.network.FSMConvResnet as FSMConvResnet

rgb_stream = FSMConvResnet.FSMresnet26(pretrained=False)

class MainNet(nn.Module):
    def __init__(self, num_classes=5):
        super(MainNet, self).__init__()
        self.num_classes = num_classes

        self.rgb_stream = rgb_stream
        self.fc = nn.Linear(10240, num_classes)
    def forward(self, rgb_data):
        # extract features
        output = self.rgb_stream(rgb_data)
        return output


