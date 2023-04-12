# Abstract model class
from abc import ABC

import torch


class Model(ABC):
    def __init__(self, num_ss_classes, model_name, repo, weights_name):
        self.num_ss_classes = num_ss_classes
        self.model_name = model_name

        # load backbone + classification layer
        self.model = torch.hub.load(repo, weights_name, pretrained=True)

        # Freeze backbone and classification layer
        for param in self.model.parameters():
            param.requires_grad = False

    # TODO: implement this!
    """@abstractmethod
    def load_ss_fc(self):
        pass"""
