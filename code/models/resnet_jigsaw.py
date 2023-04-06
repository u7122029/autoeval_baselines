import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetJigsaw(nn.Module):
    def __init__(self, num_permutations=4):
        # Use num_permutations to avoid computing the factorial.
        super(ResNetJigsaw, self).__init__()
        self.num_permutations = num_permutations

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
        self.fc_jigsaw = nn.Linear(64, self.num_permutations)

    def forward(self, x):
        x = self.feat(x)
        # flatten the feature representation
        x = x.view(x.size(0), -1)
        return self.fc(x), self.fc_jigsaw(x)