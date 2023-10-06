import torch.nn as nn

from models.model import Model


class LeNet5_SS(Model, nn.Module):
    def __init__(self, num_ss_classes=4, **kwargs):
        nn.Module.__init__(self)
        Model.__init__(self, num_ss_classes, "lenet5", "u7122029/PyTorch_CIFAR10", "lenet5",
                       **kwargs)

        # feature extraction backbone
        self.feat1 = nn.Sequential(self.model.layer1, self.model.layer2)
        self.feat2 = nn.Sequential(self.model.fc, self.model.relu, self.model.fc1, self.model.relu1)

        # classification FC layer
        self.fc = self.model.fc2

        # jigsaw prediction FC layer
        self.fc_ss = nn.Linear(self.fc.in_features, self.num_ss_classes)

    def forward(self, x):
        x = self.feat1(x)
        x = x.reshape(x.size(0), -1)
        x = self.feat2(x)

        out_class = self.fc(x)
        out_ss = self.fc_ss(x)
        return out_class, out_ss