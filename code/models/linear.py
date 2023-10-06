import torch.nn as nn

from models.model import Model


class Linear_SS(Model, nn.Module):
    def __init__(self, num_ss_classes=4, **kwargs):
        nn.Module.__init__(self)
        Model.__init__(self, num_ss_classes, "linear", "u7122029/pytorch_CIFAR10", "linear",
                       **kwargs)

        # The linear model does not have a feature extraction backbone.

        # classification FC layer
        self.fc = self.model.fc

        # jigsaw prediction FC layer
        self.fc_ss = nn.Linear(in_features=self.fc.in_features, out_features=num_ss_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out_class = self.fc(x)
        out_ss = self.fc_ss(x)
        return out_class, out_ss

if __name__ == "__main__":
    mdl = Linear_SS()
    print(mdl)