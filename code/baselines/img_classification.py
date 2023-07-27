import argparse
import os
import sys

sys.path.append(".")

import numpy as np
import torch

from eval_utils import DatasetCollection

from utils import (
    predict_multiple,
    ensure_cwd,
    normalise_path,
    TRANSFORM,
    VALID_MODELS,
    DEVICE,
    TEMP_PATH_DEFAULT,
    DATA_PATH_DEFAULT,
    DEFAULT_DATASET_COND
)

from training_utils import get_model
from pathlib import Path

parser = argparse.ArgumentParser(description="AutoEval Baselines - Image Classification")
parser.add_argument(
    "--model",
    required=True,
    type=str,
    help="The classifier to run.",
    choices=VALID_MODELS
)
parser.add_argument(
    "--train-sets",
    required=True,
    nargs="*",
    help="List of train dataset roots, space separated."
)
parser.add_argument(
    "--val-sets",
    required=True,
    nargs="*",
    help="List of validation dataset roots, space separated."
)
parser.add_argument(
    "--data-root",
    required=False,
    default=DATA_PATH_DEFAULT,
    type=str,
    help="path containing all datasets (training and validation)"
)
parser.add_argument(
    "--temp-path",
    required=False,
    default=TEMP_PATH_DEFAULT,
    type=str,
    help="The path to store temporary files."
)
parser.add_argument(
    "--recompute-acc",
    required=False,
    default=False,
    action="store_true",
    help="True if the accuracies should be recomputed. False otherwise."
)


def calculate_acc(dataloader, model, device=DEVICE):
    # function for calculating the accuracy on a given dataset
    correct = []
    for imgs, labels in iter(dataloader):
        imgs, labels = imgs.to(device), labels.to(device)
        pred, _ = predict_multiple(model, imgs)
        correct.append(pred.squeeze(1).eq(labels).cpu())
    correct = torch.cat(correct).numpy()
    return np.mean(correct)


def main(model_name,
         data_path,
         temp_file_path,
         recompute_acc,
         train_data_roots,
         val_data_roots,
         device=DEVICE):
    
    data_path = Path(data_path)
    temp_file_path = Path(temp_file_path)
    train_data_roots = [Path(i) for i in train_data_roots]
    val_data_roots = [Path(i) for i in val_data_roots]
    
    task_name = "classification"

    # load the model
    model = get_model(model_name, task_name, 4, device, load_best_fc=False)
    model.eval()

    # if there is no temp file path, make it.
    if not (temp_file_path / model_name / task_name).exists():
        (temp_file_path / model_name / task_name).mkdir()

    # need to do accuracy calculation
    predictor_func = lambda dataloader, model_m: calculate_acc(dataloader, model_m, device)
    for train_root in train_data_roots:
        dir_path = temp_file_path / model_name / task_name / train_root
        if not dir_path.exists() or recompute_acc:
            dir_path.mkdir(parents=True, exist_ok=True)
            train_data = DatasetCollection(str(train_root), data_path / train_root)
            train_accs = train_data.eval_datasets(model, TRANSFORM, predictor_func, task_name)
            np.save(str(dir_path / "train_data.npy"), train_accs)

    for val_root in val_data_roots:
        dir_path = temp_file_path / model_name / task_name / val_root
        if not dir_path.exists() or recompute_acc:
            dir_path.mkdir(parents=True, exist_ok=True)
            val_data = DatasetCollection(str(val_root), data_path / val_root)
            val_accs = val_data.eval_datasets(model, TRANSFORM, predictor_func, task_name)
            np.save(str(dir_path / "val_data.npy"), val_accs)


if __name__ == "__main__":
    ensure_cwd()
    args = parser.parse_args()
    main(args.model,
         args.data_root,
         args.temp_path,
         args.recompute_acc,
         args.train_sets,
         args.val_sets)
