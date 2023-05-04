import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model import Model


class DenseNet_SS(Model, nn.Module):
    def __init__(self, version=121, num_ss_classes=4):
        nn.Module.__init__(self)

        Model.__init__(self, num_ss_classes, "resnet", "u7122029/PyTorch_CIFAR10", f"densenet{version}")

        # feature extraction backbone
        self.feat = self.model.features

        # classification FC layer
        self.fc = self.model.classifier

        # jigsaw prediction FC layer
        # The grid has grid_length ** 2 pieces, so there are (grid_length ** 2)! permutations.
        self.fc_ss = nn.Linear(self.fc.in_features, self.num_ss_classes)

    def forward(self, x):
        x = self.feat(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        return self.fc(x), self.fc_ss(x)