import torch.nn as nn

from models.model import Model


class AlexNet_SS(Model, nn.Module):
    def __init__(self, num_ss_classes=4):
        nn.Module.__init__(self)
        Model.__init__(self, num_ss_classes, "alexnet", "u7122029/pytorch-cifar10", "alexnet")

        # feature extraction backbone
        self.feat = self.model.features
        self.feat2 = self.model.classifier[:6]

        # classification FC layer
        self.fc = self.model.classifier[-1]

        # jigsaw prediction FC layer
        self.fc_ss = nn.Linear(self.fc.in_features, self.num_ss_classes)

    def forward(self, x):
        x = self.feat(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.feat2(x)
        out_class = self.fc(x)
        out_ss = self.fc_ss(x)
        return out_class, out_ss

if __name__ == "__main__":
    model = AlexNet_SS()
    print(model)