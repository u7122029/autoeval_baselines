import torch.nn as nn
from models.model import Model


class VGG_SS(Model, nn.Module):
    def __init__(self, version, num_ss_classes=4, **kwargs):
        nn.Module.__init__(self)
        Model.__init__(self, num_ss_classes, f"vgg{version}_bn", "chenyaofo/pytorch-cifar-models",
                       f"cifar10_vgg{version}_bn", **kwargs)

        # The linear model does not have a feature extraction backbone.

        # classification FC layer
        classification_group = list(self.model.classifier.children())
        classification_layer = classification_group[-1]
        feature_last = classification_group[:-1]

        # feature extraction backbone
        self.feat = nn.Sequential(self.model.features, nn.Sequential(*feature_last))

        # classification FC layer
        self.fc = classification_layer

        # Self-supervised classification layer.
        self.fc_ss = nn.Linear(self.fc.in_features, self.num_ss_classes)

    def forward(self, x):
        # Implement the forward function exactly as how the forward function is given for the original model.
        print(x.shape)
        x = self.feat(x)
        x = x.flatten(1)
        return self.fc(x), self.fc_ss(x)


if __name__ == "__main__":
    mdl = VGG_SS(13)
    print(mdl.feat)
    print(mdl.fc)
    print(mdl.fc_ss)