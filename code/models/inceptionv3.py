import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model import Model


class Inceptionv3_SS(Model, nn.Module):
    def __init__(self, num_ss_classes=4):
        nn.Module.__init__(self)
        Model.__init__(self, num_ss_classes, "inception_v3", "u7122029/PyTorch_CIFAR10", "inception_v3")

        # feature extraction backbone
        self.feat = nn.Sequential(*list(self.model.children())[:-1])

        # classification FC layer
        self.fc = self.model.fc

        # jigsaw prediction FC layer
        self.fc_ss = nn.Linear(in_features=self.fc.in_features, out_features=num_ss_classes)

    def forward(self, x):
        x = self.feat(x)
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 2048 x 1 x 1
        x = F.dropout(x, training=False)
        # N x 2048 x 1 x 1
        x = x.view(x.size(0), -1)
        out_class = self.fc(x)
        out_ss = self.fc_ss(x)
        return out_class, out_ss
