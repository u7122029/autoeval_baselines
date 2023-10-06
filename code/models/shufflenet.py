import torch
import torch.nn as nn
from models.model import Model


class ShuffleNet_SS(Model, nn.Module):
    def __init__(self, num_ss_classes=4, **kwargs):
        nn.Module.__init__(self)
        Model.__init__(self, num_ss_classes, "shufflenet", "chenyaofo/pytorch-cifar-models",
                       "cifar10_shufflenetv2_x2_0", **kwargs)

        # feature extraction backbone
        self.feat = torch.nn.Sequential(*list(self.model.children())[:-1])

        # classification FC layer
        self.fc = list(self.model.children())[-1]

        # jigsaw prediction FC layer
        # The grid has grid_length ** 2 pieces, so there are (grid_length ** 2)! permutations.
        self.fc_ss = nn.Linear(self.fc.in_features, self.num_ss_classes)

    def forward(self, x):
        x = self.feat(x)
        x = x.mean([2, 3])
        return self.fc(x), self.fc_ss(x)
