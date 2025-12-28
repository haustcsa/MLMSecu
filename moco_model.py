import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet34, resnet50


class MoCoBase(nn.Module):
    def __init__(self, arch='resnet18'):
        super(MoCoBase, self).__init__()

        self.f = []

        if arch == 'resnet18':
            model_name = resnet18()
        elif arch == 'resnet34':
            model_name = resnet34()
        elif arch == 'resnet50':
            model_name = resnet50()
        else:
            raise NotImplementedError

        for name, module in model_name.named_children():
            if name == 'conv1':
                # 修改 conv1 层以适应新的输入方式
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        self.f = nn.Sequential(*self.f)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        return feature

class MoCo(nn.Module):
    print("进入MoCo")
    def __init__(self, feature_dim=128, arch='resnet18', momentum=0.99):
        super(MoCo, self).__init__()
        self.f = MoCoBase(arch)
        if arch == 'resnet18':
            projection_model = nn.Sequential(
                nn.Linear(512, 512, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear(512, feature_dim, bias=True)
            )
        elif arch == 'resnet34':
            projection_model = nn.Sequential(
                nn.Linear(512, 512, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear(512, feature_dim, bias=True)
            )
        elif arch == 'resnet50':
            projection_model = nn.Sequential(
                nn.Linear(2048, 512, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear(512, feature_dim, bias=True)
            )
        else:
            raise NotImplementedError

        self.g = projection_model
        self.momentum = momentum

        # 初始化动量编码器
        self.encoder_k = MoCoBase(arch)
        self.encoder_k.eval()  # 设置为评估模式，参数不会进行梯度更新
        self.g_k = projection_model
        self._update_momentum()

        #  获取特征维度
        self.dim = self._get_feature_dim()

    def _get_feature_dim(self):
        device=next(self.parameters()).device

        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            feature_q = self.f(dummy_input)
        return feature_q.size(1)

    def _update_momentum(self):
        """动量更新公式"""
        for param_q, param_k in zip(self.f.parameters(), self.encoder_k.parameters()):
            param_k.data = self.momentum * param_k.data + (1.0 - self.momentum) * param_q.data
        for param_q, param_k in zip(self.g.parameters(), self.g_k.parameters()):
            param_k.data = self.momentum * param_k.data + (1.0 - self.momentum) * param_q.data

    def forward(self, x, is_query=True):
        """前向传播"""
        if is_query:
            feature_q = self.f(x)
            out_q = self.g(feature_q)
            return F.normalize(feature_q, dim=-1), F.normalize(out_q, dim=-1)
        else:
            feature_k = self.encoder_k(x)  # 动量编码器的输出
            out_k = self.g_k(feature_k)
            return F.normalize(feature_k,dim=-1), F.normalize(out_k, dim=-1)

    def forward_query(self, x):
        """只前向通过主网络"""
        feature_q = self.f(x)
        out_q = self.g(feature_q)
        return F.normalize(feature_q, dim=-1), F.normalize(out_q, dim=-1)

    def forward_key(self, x):
        """只前向通过动量网络"""
        feature_k = self.encoder_k(x)
        out_k = self.g_k(feature_k)
        return F.normalize(feature_k, dim=-1), F.normalize(out_k, dim=-1)
