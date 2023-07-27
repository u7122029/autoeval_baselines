import os

import numpy as np
import torch
from tqdm import tqdm
from models.model import Model
from pathlib import Path

from utils import CIFAR10NP


class DatasetCollection:
    def __init__(self,
                 name: str,
                 root: Path):
        self.name = name
        self.root = root
        self.dset_paths: list[Path] = []

        self.__get_paths()

    def eval_datasets(self,
                      model: Model,
                      transform,
                      predictor_func,
                      task_name: str):
        """

        :param model:
        :param transform:
        :param predictor_func:
        :param task_name:
        :return:
        """
        acc = np.zeros(len(self))
        print(f"===> Calculating {task_name} accuracy for {self.name} using {model.model_name}")

        for i, dataloader in enumerate(tqdm(self("data.npy", "labels.npy", transform), total=len(self))):
            acc[i] = predictor_func(dataloader, model)

        return acc

    def __get_paths(self):
        """
        Gets the paths of all the datasets.
        :param filterfunc: the filter function.
        :return: Populates self.dset_paths.
        """
        for path in self.root.glob("**/data.npy"): # all parents should be unique.
            parent = path.parent
            if (parent / "labels.npy").exists():
                self.dset_paths.append(parent)

    def __len__(self):
        return len(self.dset_paths)

    def __call__(self, data_file, labels_file, transform):  # Turn this into a generator of dataloaders.
        for path in self.dset_paths:
            data_path = str(path / data_file)
            label_path = str(path / labels_file)

            dataloader = torch.utils.data.DataLoader(
                dataset=CIFAR10NP(
                    data_path=data_path,
                    label_path=label_path,
                    transform=transform,
                ),
                batch_size=500,
                shuffle=False,
            )
            yield dataloader


if __name__ == "__main__":
    d = DatasetCollection("train_data", Path("data/train_data/meta-set-group-a"))
    print(len(d.dset_paths))
