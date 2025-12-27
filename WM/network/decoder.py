import torch
import torch.nn as nn


from network.ResBlock import ResBlock
from network.Attention import ResBlock_CBAM, ChannelAttentionModule, SpatialAttentionModule
from network.simAM import Simam_module
from config import training_config as cfg
from network.ConvBlock import ConvBlock
import torch.nn.functional as F
import torch.nn.init as init


class PixelUnshuffle(nn.Module):

    def __init__(self, downscale_factor=2):
        super().__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x):

        B, C, H, W = x.shape
        r = self.downscale_factor
        assert H % r == 0 and W % r == 0, f"Height and width must be divisible by {r}"

        # [B, C, H, W] -> [B, C, H//r, r, W//r, r]
        x = x.view(B, C, H // r, r, W // r, r)
        # [B, C, r, r, H//r, W//r]
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        # [B, C * r^2, H//r, W//r]
        x = x.view(B, C * r * r, H // r, W // r)
        return x

class PerturbationClassifier(nn.Module):
    def __init__(self, in_channels=1, num_classes=4):
        super(PerturbationClassifier, self).__init__()

        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=False),

            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=False),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=False),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),

        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        # 分类器
        self.fc = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        feat = self.refine(x)

        avg_pool = self.gap(feat).view(feat.size(0), -1)
        max_pool = self.gmp(feat).view(feat.size(0), -1)
        feat_cat = torch.cat([avg_pool, max_pool], dim=1)
        out = self.fc(feat_cat)  # [B, num_classes]
        return out
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

        out = torch.cat([x3, x5], 1)

        out = self.fuse(out)

        return out


class Bottleneck1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.act1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 5, padding=2)
        self.act2 = nn.LeakyReLU(0.2)

        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.act3 = nn.LeakyReLU(0.2)
        self.conv4 = nn.Conv2d(2*out_channels, out_channels, 1)
        self.act4 = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.act1(self.conv1(x))
        out1 = self.act2(self.conv2(out))
        out2 = self.act3(self.conv3(out))
        out3 = torch.cat([out2, out1], 1)
        out4 = self.act4(self.conv4(out3))

        out = torch.cat([out4, x], 1)
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

class Decoder(nn.Module):
    def __init__(self, type, num_classes=4):
        super(Decoder, self).__init__()
        self.type = type

        if self.type == "tracer":
            self.in_channels = 6
            self.classifier = None  # tracer 模式不需要分类器
        elif self.type == "detector":
            self.in_channels = 3
            self.classifier = PerturbationClassifier(in_channels=64, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown decoder type: {self.type}")
        self.attention = ResBlock_CBAM(64, 16)
        self.feature_extractor1 = spatial_features_extractor(in_c=self.in_channels+48)
        self.simam = Simam_module()
        self.skip_proj = nn.Sequential(
            nn.Conv2d(64, 256, 3,padding=1),
            nn.LeakyReLU(0.2)
        )

        self.b1 = Bottleneck(64, 64)
        self.b2 = Bottleneck1(128, 64)
        self.b3 = Bottleneck2(192, 64)
        self.down = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 64, 3, padding=1),
            #nn.LeakyReLU(0.2),

            nn.Conv2d(64, 32, 3, padding=1),
            #nn.LeakyReLU(0.2),

            nn.Conv2d(32, cfg.wm_channels, 3, padding=1),
        )
        self.unshuffle = PixelUnshuffle(4)
        self.alpha = nn.Parameter(torch.tensor(0.1))

        self.fc = nn.Linear(4096, cfg.message_length)

    def forward(self, x,aa=None,diff=None):
        diff = self.unshuffle(diff)
        x1=torch.cat([diff, x], 1)


        fm = self.feature_extractor1(x1)
        fm=self.attention(fm)
        skip = self.skip_proj(fm)

        o = self.b1(fm)
        o = o + self.simam(o)
        o = self.b2(o)
        o = o + self.simam(o)
        o = self.b3(o)
        o = o + self.simam(o)

        o = o + skip


        message = self.down(o)
        message = message.squeeze(1).view(message.size(0), -1)
        message = self.fc(message)

        # 只有 detector 才输出分类
        if self.type == "detector":
            perturb_pred = self.classifier(fm)
        else:
            perturb_pred = None

        return message, perturb_pred
