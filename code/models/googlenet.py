import torch.nn as nn

from models.model import Model


class GoogLeNet_SS(Model, nn.Module):
    def __init__(self, num_ss_classes=4, **kwargs):
        nn.Module.__init__(self)
        Model.__init__(self, num_ss_classes, "googlenet", "u7122029/PyTorch_CIFAR10", "googlenet",
                       **kwargs)

        # feature extraction backbone
        self.feat = nn.Sequential(*list(self.model.children())[:-1])

        # classification FC layer
        self.fc = self.model.fc

        # jigsaw prediction FC layer
        self.fc_ss = nn.Linear(self.fc.in_features, self.num_ss_classes)

    def forward(self, x):
        x = self.feat(x)
        x = x.reshape(x.size(0), -1)
        out_class = self.fc(x)
        out_ss = self.fc_ss(x)
        return out_class, out_ss


if __name__ == "__main__":
    import torch
    model = GoogLeNet_SS()
    print(model(torch.zeros(100,3,32,32)))
