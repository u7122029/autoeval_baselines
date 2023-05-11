import torch
import torch.nn as nn

from models.model import Model


class OBC_SS(Model, nn.Module):
    def __init__(self, num_ss_classes=4, device="cuda"):
        nn.Module.__init__(self)
        Model.__init__(self, num_ss_classes, "obc", "u7122029/pytorch-cifar10", "obc")
        self.device = device

        # The OBC model does not have a feature extraction backbone.

        # classification FC layer
        self.fc = self.model.fc

        # jigsaw prediction FC layer
        self.fc_ss = nn.Linear(in_features=self.fc.in_features, out_features=num_ss_classes)

    def forward(self, x):
        x = torch.mean(x, [1, 2, 3]).reshape(-1, 1).to(self.device)
        out_class = self.fc(x)
        out_ss = self.fc_ss(x)
        return out_class, out_ss
