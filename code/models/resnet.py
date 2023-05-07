import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model import Model


class ResNet_SS(Model, nn.Module):
    def __init__(self, version=56, num_ss_classes=4):
        nn.Module.__init__(self)
        if version in {110, 1202}:
            Model.__init__(self, num_ss_classes, f"resnet{version}", "u7122029/pytorch_resnet_cifar10", f"resnet{version}")
        else:
            Model.__init__(self, num_ss_classes, f"resnet{version}", "chenyaofo/pytorch-cifar-models", f"cifar10_resnet{version}")

        self.version = version

        # feature extraction backbone
        self.feat = torch.nn.Sequential(*list(self.model.children())[:-1])

        # classification FC layer
        self.fc = list(self.model.children())[-1]

        # jigsaw prediction FC layer
        # The grid has grid_length ** 2 pieces, so there are (grid_length ** 2)! permutations.
        self.fc_ss = nn.Linear(self.fc.in_features, self.num_ss_classes)


    def forward(self, x):
        # Implement the forward function exactly as how the forward function is given for the original model.
        x = self.feat(x)
        if self.version in {110, 1202}:
            x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)
        return self.fc(x), self.fc_ss(x)
