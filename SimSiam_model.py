import torch
import torch.nn as nn

class SimSiamBase(nn.Module):
    """
    Base feature extractor module supporting ResNet18.
    """
    def __init__(self):
        super(SimSiamBase, self).__init__()
        self.f = []
        from torchvision.models import resnet18

        for name, module in resnet18(zero_init_residual=True).named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        self.f = nn.Sequential(*self.f)

    def forward(self, x):
        return self.f(x)


class SimSiam(nn.Module):
    """
    SimSiam model with projection head and optional perturbations.
    """
    def __init__(self, feature_dim=2048, delta_size=1, delta_weight=0.1, classwise=False):
        super(SimSiam, self).__init__()
        self.f = SimSiamBase()

        # Define projection head
        self.projection_head = nn.Sequential(
            nn.Linear(512, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, feature_dim, bias=False),
            nn.BatchNorm1d(feature_dim)
        )

        # Initialize perturbations
        self.initialize_delta(delta_size, delta_weight, classwise)

    def initialize_delta(self, delta_size, delta_weight, classwise):
        if classwise:
            print("Classwise perturbation initialization")
            self.delta = nn.Parameter(torch.randn(delta_size, 3, 32, 32) * delta_weight)
        else:
            print("Samplewise perturbation initialization")
            self.delta = nn.Parameter(torch.randn(1, 3, 32, 32) * delta_weight)

    def forward(self, x, labels=None):
        if self.delta is not None:
            if labels is not None and len(labels) > 0:
                x = torch.clamp(x + self.delta[labels], min=0., max=1.)
            else:
                x = torch.clamp(x + self.delta[0], min=0., max=1.)

        # Extract features
        feature = torch.flatten(self.encoder(x), start_dim=1)

        # Pass features through projection head
        proj = self.projection_head(feature)

        return feature, proj



    # def forward(self, x, labels=None):
    #     feature = torch.flatten(self.encoder(x), start_dim=1)
    #     proj = self.projection_head(feature)
    #     return feature, proj




