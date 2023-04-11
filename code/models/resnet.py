import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model import Model

class ResNet_SS(Model, nn.Module):
    def __init__(self, num_ss_classes):
        # TODO: Allow other versions of resnet to be used.
        nn.Module.__init__(self)
        Model.__init__(self, num_ss_classes, "resnet", "chenyaofo/pytorch-cifar-models", "cifar10_resnet56")

        # feature extraction backbone
        self.feat = torch.nn.Sequential(*list(self.model.children())[:-1])

        # classification FC layer
        self.fc = list(self.model.children())[-1]

        # jigsaw prediction FC layer
        # The grid has grid_length ** 2 pieces, so there are (grid_length ** 2)! permutations.
        self.fc_ss = nn.Linear(64, self.num_ss_classes)

    def forward(self, x):
        x = self.feat(x)
        # flatten the feature representation
        x = x.view(x.size(0), -1)
        return self.fc(x), self.fc_ss(x)