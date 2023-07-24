import argparse
import os
import sys

sys.path.append(".")

import numpy as np
import torch

from eval_utils import (
    eval_train,
    eval_validation
)

from utils import (
    predict_multiple,
    get_dirs,
    TRANSFORM,
    VALID_MODELS,
    DEVICE,
    TEMP_PATH_DEFAULT,
    DATA_PATH_DEFAULT,
    TRAIN_DATA, VAL_DATA,
)
from training_utils import get_model

parser = argparse.ArgumentParser(description="AutoEval Baselines - Image Classification")
parser.add_argument(
    "--model",
    required=True,
    type=str,
    help="The classifier to run.",
    choices=VALID_MODELS
)
parser.add_argument(
    "--dataset-path",
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


def main(model_name, dataset_path, temp_file_path, recompute_acc, train_set, val_sets,
         device=DEVICE):
    if dataset_path.endswith('/'):
        dataset_path = dataset_path[:-1]

    batch_size = 500
    # load the model
    model = get_model(model_name, "accuracy", 4, device, load_best_fc=False)
    model.eval()

    # if there is no temp file path, make it.
    if not os.path.exists(f"{temp_file_path}/{model_name}/classification"):
        os.makedirs(f"{temp_file_path}/{model_name}/classification")

    # need to do accuracy calculation
    predictor_func = lambda dataloader: calculate_acc(dataloader, model, device)
    if not os.path.exists(f"{temp_file_path}/{model_name}/classification/{train_set}.npy") or recompute_acc:
        eval_train(
            dataset_path,
            temp_file_path,
            train_set,
            TRANSFORM,
            batch_size,
            predictor_func,
            "classification",
            model_name
        )

    if not os.path.exists(f"{temp_file_path}/{model_name}/classification/val_sets.npy") or recompute_acc:
        eval_validation(
            temp_file_path,
            val_sets,
            TRANSFORM,
            batch_size,
            predictor_func,
            "classification",
            model_name
        )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.model,
         args.dataset_path,
         args.temp_path,
         args.recompute_acc,
         TRAIN_DATA,
         get_dirs(VAL_DATA, DATA_PATH_DEFAULT))
