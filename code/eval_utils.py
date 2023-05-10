import os

import numpy as np
import torch
from tqdm import tqdm

from utils import CIFAR10NP


def eval_train(dataset_path, temp_file_path, train_set, transform, batch_size, predictor_func,
               task_name="self-supervision"):
    """
    :param dataset_path:
    :param temp_file_path:
    :param train_set:
    :param transform:
    :param batch_size:
    :param predictor_func: The predictor for the given task.
    :return:
    """
    # Load the training set data.
    train_path = f"{dataset_path}/{train_set}"
    train_candidates = []
    for file in sorted(os.listdir(train_path)):
        if file.endswith(".npy") and file.startswith("new_data"):
            train_candidates.append(file)

    print(f"===> Calculating {task_name} accuracy for {train_set}")
    acc = np.zeros(len(train_candidates))
    for i, candidate in enumerate(tqdm(train_candidates)):
        data_path = f"{train_path}/{candidate}"
        label_path = f"{train_path}/labels.npy"

        train_dataloader = torch.utils.data.DataLoader(
            dataset=CIFAR10NP(
                data_path=data_path,
                label_path=label_path,
                transform=transform,
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        acc[i] = predictor_func(train_dataloader)

    np.save(f"{temp_file_path}{train_set}.npy", acc)


def eval_validation(dataset_path, temp_file_path, val_sets, transform, batch_size, predictor_func,
                    task_name="self-supervision", save_results=True):
    """

    :param dataset_path:
    :param val_sets:
    :param transform:
    :param batch_size:
    :param predictor_func:
    :return:
    """
    # load validation set.
    val_candidates = []
    val_paths = [f"{dataset_path}/{set_name}" for set_name in val_sets]
    for val_path in val_paths:
        for file in sorted(os.listdir(val_path)):
            val_candidates.append(f"{val_path}/{file}")

    acc = np.zeros(len(val_candidates))
    print(f"===> Calculating {task_name} accuracy for validation sets")

    for i, candidate in enumerate(tqdm(val_candidates)):
        data_path = f"{candidate}/data.npy"
        label_path = f"{candidate}/labels.npy"

        dataloader = torch.utils.data.DataLoader(
            dataset=CIFAR10NP(
                data_path=data_path,
                label_path=label_path,
                transform=transform,
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        acc[i] = predictor_func(dataloader)
    if save_results:
        np.save(f"{temp_file_path}val_sets.npy", acc)
