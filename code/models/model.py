# Abstract model class
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

class Model(ABC, nn.Module):
    def __init__(self, num_ss_classes, model_name, repo, weights_name):
        super().__init__()
        self.num_ss_classes = num_ss_classes
        self.model_name = model_name

        # load backbone + classification layer
        self.model = torch.hub.load(repo, weights_name, pretrained=True)

        # Freeze backbone and classification layer
        for param in self.model.parameters():
            param.requires_grad = False

        self.fc_ss = None # Every model needs this self-supervised layer, unless it is only being used for class acc.


    def load_ss_fc(self, link, is_local=False):
        # Load the weights of the self-supervised fc layer.
        if is_local:
            pretrained_state_dict = torch.load(link, map_location="cpu")
        else:
            pretrained_state_dict = torch.hub.load_state_dict_from_url(link, map_location="cpu")

        our_state_dict = self.state_dict()
        if "fc_ss.1.weight" in pretrained_state_dict:
            our_state_dict["fc_ss.1.weight"] = pretrained_state_dict["fc_ss.1.weight"]
            our_state_dict["fc_ss.1.bias"] = pretrained_state_dict["fc_ss.1.bias"]
        else:
            our_state_dict["fc_ss.weight"] = pretrained_state_dict["fc_ss.weight"]
            our_state_dict["fc_ss.bias"] = pretrained_state_dict["fc_ss.bias"]

