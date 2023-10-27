# Abstract model class
import os
from abc import ABC

import torch
import torch.nn as nn


class Model(ABC, nn.Module):
    def __init__(self, num_ss_classes, model_name, repo, weights_name, **kwargs):
        """
        Constructor for Generic Model
        :param num_ss_classes: The number of self-supervised classes, eg: 4 for rotation prediction.
        :param model_name: The name of the model.
        :param repo: The repository to download the weights.
        :param weights_name: The name of the weights.
        :param kwargs: Includes extra arguments for the torch.hub.load function such as
        force_reload (bool), pretrained (bool) and dataset (mnist or cifar10).
        """
        super().__init__()
        self.num_ss_classes = num_ss_classes
        self.model_name = model_name

        # load backbone + classification layer
        self.model = torch.hub.load(repo, weights_name, **kwargs)

        # Freeze backbone and classification layer
        self.freeze_backbone()

        self.fc_ss = None  # Every model needs this self-supervised layer, unless it is only being used for class classification.

    def freeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def load_ss_fc(self, link, is_local=False):
        # Load the weights of the self-supervised fc layer.
        if is_local:
            self.fc_ss.load_state_dict(torch.load(link, map_location="cpu"))
            return
        self.fc_ss.load_state_dict(torch.hub.load_state_dict_from_url(link, map_location="cpu"))

    def save_ss_fc(self, out_dir, filename):
        """
        Saves the state dictionary of the self-supervised fully connected layer to the given output directory.
        :param out_dir: The output directory.
        :param filename: The name of the output file.
        :return: True if the state dict was successfully saved, and false otherwise.
        """
        if not self.fc_ss:
            return False

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        fc_state_dict = self.fc_ss.state_dict()
        torch.save(fc_state_dict, f"{out_dir}/{filename}")
        return True
