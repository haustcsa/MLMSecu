import torch
import torch.nn as nn
import torch.nn.functional as F

class DownsampleConv(nn.Module):
    def __init__(self, in_channels):
        super(DownsampleConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2)

    def forward(self, X_s, X_f):
        X_s = self.conv(X_s)
        X_f = self.conv(X_f)
        out = X_s + X_f
        return out


class SpatialPyramidPooling(nn.Module):
    def __init__(self):
        super(SpatialPyramidPooling, self).__init__()
        self.pool1x1 = nn.AdaptiveAvgPool2d((1, 1))
        self.pool4x4 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.pool2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):

        b, c, h, w = x.size()
        out_1x1 = self.pool1x1(x)
        out_1x1 = out_1x1.view(b, -1)

        out_4x4 = self.pool4x4(x)
        out_4x4 = out_4x4.view(b, -1)

        spp_out = torch.cat([out_1x1, out_4x4], dim=1)
        return spp_out