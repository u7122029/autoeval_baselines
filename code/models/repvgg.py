import torch
import torch.nn as nn

from models.model import Model


class RepVGG_SS(Model, nn.Module):
    """
    num_ss_classes: number of self supervised task classes. 4 by default for rotation.
    """

    def __init__(self, num_ss_classes=4, **kwargs):
        nn.Module.__init__(self)
        Model.__init__(self, num_ss_classes, "repvgg", "chenyaofo/pytorch-cifar-models",
                       "cifar10_repvgg_a0", **kwargs)

        # feature extraction backbone
        # automatically frozen in parent class.
        self.feat = torch.nn.Sequential(*list(self.model.children())[:-1])

        # classification FC layer
        # parent class automatically freezes this since it comes from self.model.
        self.fc = list(self.model.children())[-1]

        # rotation prediction FC layer
        # not frozen since this is a completely new layer.
        self.fc_ss = nn.Linear(1280, self.num_ss_classes)

    def forward(self, x):
        x = self.feat(x)
        # flatten the feature representation
        x = x.view(x.size(0), -1)
        return self.fc(x), self.fc_ss(x)
