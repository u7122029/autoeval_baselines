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

from utils import predict_multiple, TRANSFORM, valid_models
from training_utils import get_model

parser = argparse.ArgumentParser(description="AutoEval baselines - get_accuracy")
parser.add_argument(
    "--model", required=True, type=str, help="the model used to run this script"
)
parser.add_argument(
    "--dataset-path",
    required=False,
    default="data",
    type=str,
    help="path containing all datasets (training and validation)",
    choices=valid_models
)
parser.add_argument(
    "--compute-acc",
    required=False,
    default=False,
    action="store_true",
    help="True if the accuracies should be recomputed. False otherwise."
)


def calculate_acc(dataloader, model, device):
    # function for calculating the accuracy on a given dataset
    correct = []
    for imgs, labels in iter(dataloader):
        imgs, labels = imgs.to(device), labels.to(device)
        pred, _ = predict_multiple(model, imgs)
        correct.append(pred.squeeze(1).eq(labels).cpu())
    correct = torch.cat(correct).numpy()
    return np.mean(correct)


if __name__ == "__main__":
    # paths
    args = parser.parse_args()
    dataset_path = args.dataset_path
    if not dataset_path or dataset_path[-1] != "/":
        dataset_path += "/"

    model_name = args.model
    train_set = "train_data"
    val_sets = sorted(["cifar10-f-32", "cifar-10.1-c", "cifar-10.1"])
    temp_file_path = f"../temp/{model_name}/acc/"

    batch_size = 500
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load the model
    model = get_model(model_name, "accuracy", 4, device, load_best_fc=False)
    model.eval()

    # if there is no temp file path, make it.
    if not os.path.exists(temp_file_path):
        os.makedirs(temp_file_path)

    # need to do accuracy calculation
    predictor_func = lambda dataloader: calculate_acc(dataloader, model, device)
    if not os.path.exists(f"{temp_file_path}{train_set}.npy") or args.compute_acc:
        eval_train(
            dataset_path,
            temp_file_path,
            train_set,
            TRANSFORM,
            batch_size,
            predictor_func,
            task_name="classification"
        )

    if not os.path.exists(f"{temp_file_path}val_sets.npy") or args.compute_acc:
        eval_validation(
            dataset_path,
            temp_file_path,
            val_sets,
            TRANSFORM,
            batch_size,
            predictor_func,
            task_name="classification"
        )
