# Helper class for having ensemble models.

import torch
import torch.nn as nn
import torch.nn.functional as F
import network


class GeneratorEnsemble(nn.Module):
    """Wrapper modeule for an ensemble of generator models"""

    def __init__(self, nz: 256, num_channels: 3, img_size: 32, G_activation: torch.tanh, grayscale: 0,
                 ensemble_size: 2):
        super(GeneratorEnsemble, self).__init__()

        self.subnets = nn.ModuleList(
            [network.gan.GeneratorA(nz=nz, nc=num_channels, img_size=32, activation=G_activation,
                                    grayscale=grayscale) for i in range(ensemble_size)])
        # self.subnets = nn.ModuleList(
        #     [network.gan.DCGAN(nz=nz, nc=num_channels,
        #                             grayscale=grayscale) for i in range(ensemble_size)])

    def forward(self, x, idx: int = -1):
        if idx >= 0:
            return self.subnets[idx].forward(x)
        results = []
        for i in range(len(self.subnets)):
            results.append(self.subnets[i].forward(x))
        return torch.stack(results, dim=1)

    def variance(self, x):
        results = []
        with torch.no_grad():
            for i in range(len(self.subnets)):
                results.append(self.subnets[i].forward(x))
            return torch.var(F.softmax(torch.stack(results, dim=1), dim=-1), dim=1)

    def size(self):
        return len(self.subnets)

    def get_model_by_idx(self, idx):
        return self.subnets[idx]
