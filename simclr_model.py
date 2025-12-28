import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet34, resnet50


class SimCLRBase(nn.Module):

    def __init__(self, arch='resnet18'):
        super(SimCLRBase, self).__init__()
        self.f = []
        if arch == 'resnet18':
            print("resnet18")
            model_name = resnet18()
        elif arch == 'resnet34':
            print("resnet34")
            model_name = resnet34()
        elif arch == 'resnet50':
            print("resnet50")
            model_name = resnet50()
        else:
            raise NotImplementedError

        for name, module in model_name.named_children():
            if name == 'conv1':
                print("simclr_model")
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        self.f = nn.Sequential(*self.f)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)

        return feature


class SimCLR(nn.Module):
    def __init__(self, feature_dim=128, arch='resnet18',delta_size=1,delta_weight=0.1,classwise=False):
        super(SimCLR, self).__init__()

        self.f = SimCLRBase(arch)
        if arch == 'resnet18':
            print("resnet18")
            projection_model = nn.Sequential(nn.Linear(512, 512, bias=False),
                                             nn.BatchNorm1d(512),
                                             nn.ReLU(inplace=True),
                                             nn.Linear(512, feature_dim, bias=True))
        elif arch == 'resnet34':
            print("resnet34")
            projection_model = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512),
                                             nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
        elif arch == 'resnet50':
            print("resnet50")
            projection_model = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                                             nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
        else:
            raise NotImplementedError

        self.g = projection_model
        # xtj delta
        self.initialize_delta(delta_size,delta_weight,classwise)

    def initialize_delta(self, delta_size, delta_weight, classwise):
        if classwise:
            print("进入classwise")
            self.delta = nn.Parameter(torch.randn(delta_size, 3, 32, 32) * delta_weight)
        else:
            print("进入samplewise")
            self.delta = nn.Parameter(torch.randn(1, 3, 32, 32) * delta_weight)

    def forward(self, x, index=None, labels=None):
        if self.delta is not None:
            if labels is not None and len(labels) > 0:
                x = torch.clamp(x + self.delta[labels], min=0., max=1.)
            else:
                x = torch.clamp(x + self.delta[0], min=0., max=1.)
                feature = self.f(x)
                out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)

    # def forward(self, x):
    #     feature = self.f(x)
    #     out = self.g(feature)
    #     return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
