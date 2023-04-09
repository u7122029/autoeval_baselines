import torch
import torch.nn as nn
import torch.nn.functional as F


class MobileNet_SS(nn.Module):
    def __init__(self, num_ss_classes=4):
        # Use num_permutations to avoid computing the factorial.
        super(MobileNet_SS, self).__init__()
        self.num_ss_classes = num_ss_classes
        self.model_name = "mobilenetv2"

        # load the pretrained model weight
        # these feature extraction backbone parameters are freezed
        # shouldn't be changed during rotation prediction training
        self.model = torch.hub.load(
            "chenyaofo/pytorch-cifar-models", f"cifar10_mobilenetv2_x1_4", pretrained=True
        )

        # feature extraction backbone
        self.feat = self.model.features

        # classification FC layer
        self.fc = self.model.classifier

        # jigsaw prediction FC layer
        self.fc_ss = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(1792, self.num_ss_classes)
        )
    def forward(self, x):
        x = self.feat(x)
        # flatten the feature representation
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        out_class = self.fc(x)
        out_ss = self.fc_ss(x)
        return out_class, out_ss