import argparse
from tqdm import tqdm
import sys

sys.path.append(".")

import numpy as np
import torch

from utils import (
    dataset_recurse,
    predict_multiple,
    ensure_cwd,
    VALID_MODELS,
    DEVICE,
    TEMP_PATH_DEFAULT,
    DATA_PATH_DEFAULT,
    RESULTS_PATH_DEFAULT
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
    "--dsets",
    required=True,
    nargs="*",
    help="List of relative train dataset roots, space separated."
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
    "--results-path",
    required=False,
    default=RESULTS_PATH_DEFAULT,
    type=str,
    help="The path to store results."
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
         data_root,
         temp_file_path,
         results_path,
         dset_paths,
         device=DEVICE):
    data_root = Path(data_root)
    temp_file_path = Path(temp_file_path)
    results_path = Path(results_path)
    dset_paths = [Path(i) for i in dset_paths]

    task_name = "classification"

    # load the model
    model = get_model(model_name, task_name, 4, device, load_best_fc=False)
    model.eval()

    # Classification accuracy predictor.
    predictor_func = lambda dataloader, model_m: calculate_acc(dataloader, model_m, device)
    for dset_collection_root in dset_paths:
        results_root = results_path / "raw_findings" / dset_collection_root
        dataset_recurse(data_root / dset_collection_root, results_root, task_name, model, predictor_func)


if __name__ == "__main__":
    ensure_cwd()
    args = parser.parse_args()
    main(args.model,
         args.data_root,
         args.temp_path,
         args.results_path,
         args.dsets)
