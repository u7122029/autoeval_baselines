import argparse
import sys

sys.path.append(".")

import numpy as np
import torch

from utils import (
    generate_results,
    predict_multiple,
    ensure_cwd,
    VALID_MODELS,
    DEVICE,
    TEMP_PATH_DEFAULT,
    DATA_PATH_DEFAULT,
    RESULTS_PATH_DEFAULT,
    VALID_DATASETS
)

parser = argparse.ArgumentParser(description="AutoEval Baselines - Image Classification")
parser.add_argument(
    "--model",
    required=True,
    type=str,
    help="The classifier to run.",
    choices=VALID_MODELS
)
parser.add_argument(
    "--dataset",
    required=False,
    default="cifar10",
    help="The name of the dataset that should be used.",
    choices=VALID_DATASETS
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
    "--results-path",
    required=False,
    default=RESULTS_PATH_DEFAULT,
    type=str,
    help="The path to store results."
)
parser.add_argument(
    "--recalculate-results",
    action="store_true",
    required=False,
    help="Whether the task should be recalculated over the given dataset paths."
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


def main(*ags, **kwargs):
    generate_results(*ags, **kwargs, load_best_fc=False)


if __name__ == "__main__":
    ensure_cwd()
    args = parser.parse_args()
    main(args.dataset,
         args.model,
         "classification",
         args.data_root,
         args.results_path,
         args.dsets,
         4,
         calculate_acc,
         recalculate_results=args.recalculate_results)
