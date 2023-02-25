import torch
import torch.nn as nn
from math import factorial


class ResNetJigsaw(nn.Module):
    def __init__(self, grid_length=2):
        super(ResNetJigsaw, self).__init__()
        self.grid_length = grid_length
        # load the pretrained model weight
        # these feature extraction backbone parameters are freezed
        # shouldn't be changed during rotation prediction training
        self.model = torch.hub.load(
            "chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True
        )
        # feature extraction backbone
        self.feat = torch.nn.Sequential(*list(self.model.children())[:-1])
        # classification FC layer
        self.fc = list(self.model.children())[-1]

        # jigsaw prediction FC layer
        # The grid has grid_length ** 2 pieces, so there are (grid_length ** 2)! permutations.
        self.fc_permutation = nn.Linear(64, factorial(self.grid_length ** 2))

    def forward(self, x):
        x = self.feat(x)
        # flatten the feature representation
        x = x.view(x.size(0), -1)
        return self.fc(x), self.fc_permutation(x)