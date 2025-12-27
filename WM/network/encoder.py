import torch
import torch.nn as nn
import torch.nn.functional as F
from network.Attention1 import ResBlock_CBAM
from network.ConvBlock import ConvBlock
from network.simAM import Simam_module
from config import training_config as cfg
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchsummary import summary
import bchlib


plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用 SimHei 字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class spatial_features_extractor(nn.Module):
    def __init__(self, in_c=3):
        super().__init__()
        self.conv3 = nn.Conv2d(in_c, 32, 3, padding=1)
        self.conv5 = nn.Conv2d(in_c, 32, 7, padding=3)

        self.shortcut = nn.Conv2d(in_c, 64, 1)

        self.fuse = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):

        x3 = F.leaky_relu(self.conv3(x), 0.2)
        x5 = F.leaky_relu(self.conv5(x), 0.2)

        x = torch.cat([x3, x5], 1)

        out = self.fuse(x)

        return out

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 4 * out_channels, 1)
        self.act1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(4 * out_channels, 2 * out_channels, 3, padding=1)
        self.act2 = nn.LeakyReLU(0.2)

        self.conv3 = nn.Conv2d(2 * out_channels, out_channels, 3, padding=1)
        self.act3 = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.act1(self.conv1(x))
        out = self.act2(self.conv2(out))
        out = self.act3(self.conv3(out))
        out = torch.cat([out, x], 1)
        return out


class Bottleneck2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bottleneck2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, 1)
        self.act1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.act2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(128, out_channels, 3, padding=1)
        self.act3 = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.act1(self.conv1(x))
        out = self.act2(self.conv2(out))
        out = self.act3(self.conv3(out))
        out = torch.cat([out, x], 1)
        return out
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.attention = ResBlock_CBAM(64, 16)
        self.simam = Simam_module()
        self.linear0 = nn.Linear(cfg.message_length, 4 * 32 * 32)
        self.linear1 = nn.Linear(cfg.message_length, 4 * 32 * 32)
        self.linear2 = nn.Linear(cfg.message_length, 4 * 32 * 32)

        wm_channels = [cfg.wm_channels, cfg.wm_channels, cfg.wm_channels]
        self.conv_wm_0 = ConvBlock(4, wm_channels[0], blocks=2)
        self.conv_wm_1 = ConvBlock(4, wm_channels[1], blocks=2)
        self.conv_wm_2 = ConvBlock(4, wm_channels[2], blocks=2)

        self.feature_extractor = spatial_features_extractor()
        self.b0 = Bottleneck(64 + wm_channels[0], 64)
        self.b1 = Bottleneck(64 * 2 + wm_channels[0] + wm_channels[1], 64)
        self.b2 = Bottleneck2(64 * 3 + wm_channels[0] + wm_channels[1] + wm_channels[2], 64)

        self.down = nn.Sequential(
            nn.Conv2d(64 * 3 + wm_channels[0] + wm_channels[1] + wm_channels[2] + 64, 128, 1),
            #nn.LeakyReLU(0.2),

            nn.Conv2d(128, 64, 1),
            #nn.LeakyReLU(0.2),

            nn.Conv2d(64, 32, 1),
            #nn.LeakyReLU(0.2),

            nn.Conv2d(32, 3, 1),
            nn.Tanh(),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0.2, mode='fan_in',nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        for linear_layer in [self.linear0, self.linear1, self.linear2]:
            init.xavier_uniform_(linear_layer.weight)
            init.constant_(linear_layer.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
                torch.nn.init.constant_(m.running_mean, 0)
                torch.nn.init.constant_(m.running_var, 1)

    def forward(self, x, watermark, factor):

        fm = self.feature_extractor(x)
        fm = self.attention(fm)
        wm_ex_0 = self.linear0(watermark)

        wm_ex = wm_ex_0.view(-1, 4, 32, 32)

        wm_ex_0 = self.conv_wm_0(wm_ex)
        o = self.b0(torch.cat([fm, wm_ex_0], dim=1))
        o = o + self.simam(o)

        wm_ex_1 = self.conv_wm_1(wm_ex)
        o = self.b1(torch.cat([o, wm_ex_1], dim=1))
        o = o + self.simam(o)


        wm_ex_2 = self.conv_wm_2(wm_ex)
        o = self.b2(torch.cat([o, wm_ex_2], dim=1))
        o = o + self.simam(o)

        wm_diff = self.down(o)

        out = x + factor * wm_diff

        return out

