import torch
import torch.nn as nn


class RepVGG_SS(nn.Module):
    """
    num_ss_classes: number of self supervised task classes. 4 by default for rotation.
    """
    def __init__(self, num_ss_classes=4):
        super(RepVGG_SS, self).__init__()
        self.num_ss_classes = num_ss_classes
        self.model_name = "repvgg"

        # load the pretrained model weight
        # these feature extraction backbone parameters are freezed
        # shouldn't be changed during rotation prediction training
        self.model = torch.hub.load(
            "chenyaofo/pytorch-cifar-models", "cifar10_repvgg_a0", pretrained=True
        )
        # feature extraction backbone
        self.feat = torch.nn.Sequential(*list(self.model.children())[:-1])
        # classification FC layer
        self.fc = list(self.model.children())[-1]
        # rotation prediction FC layer
        self.fc_ss = nn.Linear(1280, self.num_ss_classes)

    def forward(self, x):
        x = self.feat(x)
        # flatten the feature representation
        x = x.view(x.size(0), -1)
        return self.fc(x), self.fc_ss(x)
