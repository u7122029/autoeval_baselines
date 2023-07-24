import os

import numpy as np
import torch
from tqdm import tqdm

from utils import CIFAR10NP


def eval_train(dataset_path, temp_file_path, train_set, transform, batch_size, predictor_func, task_name, model_name,
               save_results=True):
    """
    TODO: Add documentation.
    @param dataset_path:
    @param temp_file_path:
    @param train_set:
    @param transform:
    @param batch_size:
    @param predictor_func:
    @param task_name:
    @param model_name:
    @param save_results:
    @return:
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

    if save_results:
        np.save(f"{temp_file_path}/{model_name}/{task_name}/{train_set}.npy", acc)


def eval_validation(temp_file_path, val_sets, transform, batch_size, predictor_func,
                    task_name, model_name, save_results=True):
    """
    TODO: Add documentation.
    @param temp_file_path:
    @param val_sets:
    @param transform:
    @param batch_size:
    @param predictor_func:
    @param task_name:
    @param model_name:
    @param save_results:
    @return:
    """
    acc = np.zeros(len(val_sets))
    print(f"===> Calculating {task_name} accuracy for validation sets")

    for i, candidate in enumerate(tqdm(val_sets)):
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
        np.save(f"{temp_file_path}/{model_name}/{task_name}/val_sets.npy", acc)
