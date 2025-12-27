
import torch
import torch.nn as nn
import torch.nn.functional as F

from network.Attention import ResBlock_CBAM


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)

        attn = self.mlp(avg_pool) + self.mlp(max_pool)
        attn = self.sigmoid(attn).view(b, c, 1, 1)
        return attn


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg, max], dim=1)
        attn = self.sigmoid(self.conv(x_cat))
        return attn


class CBAM(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.ca = ChannelAttentionModule(channel)
        self.sa = SpatialAttentionModule()

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + x

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k):
        super().__init__()
        padding = k // 2
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, padding=padding)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(self.conv(x))

class MultiScaleBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv3 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.conv5 = nn.Conv2d(in_c, out_c, 5, padding=2)
        self.fuse = nn.Conv2d(out_c*2, out_c, 1)

    def forward(self, x):
        x3 = F.leaky_relu(self.conv3(x), 0.2)
        x5 = F.leaky_relu(self.conv5(x), 0.2)

        x = torch.cat([x3, x5], 1)
        x = F.leaky_relu(self.fuse(x), 0.2)
        return x
class DifferencePredictor(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 1),
            nn.LeakyReLU(0.2)
        )

        self.block1 = MultiScaleBlock(32, 64)
        self.cbam1 = ResBlock_CBAM(64, 16)
        self.block2 = MultiScaleBlock(64, 64)
        self.cbam2 = ResBlock_CBAM(64, 16)

        self.decoder = nn.Sequential(
            nn.Conv2d(64+3, 32, 3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 16, 3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(16, 3, 3, padding=1)
        )

    def forward(self, x1):
        x = self.stem(x1)
        x = self.block1(x)
        x = self.cbam1(x)
        x = self.block2(x)
        x = self.cbam2(x)
        x = torch.cat([x1, x], 1)
        out = self.decoder(x)
        return out


class DifferenceLearner(nn.Module):
    def __init__(self):
        super().__init__()
        self.predictor = DifferencePredictor()

    def forward(self, attacked, original=None, use_pred=False):
        pred_diff = self.predictor(attacked)

        diff_loss = None

        if original is not None:
            true_diff = original - attacked[:, 2:3, :, :]

            diff_loss = F.mse_loss(pred_diff[:, 2:3, :, :], true_diff)

            if use_pred:
                used_diff = pred_diff
            else:
                used_diff = pred_diff
        else:
            used_diff = pred_diff

        return used_diff, diff_loss