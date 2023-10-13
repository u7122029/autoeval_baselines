import argparse
import sys
sys.path.append(".")

import numpy as np
import torch
from pathlib import Path

from utils import (
    ensure_cwd,
    VALID_MODELS,
    VALID_DATASETS,
    DEVICE,
    RESULTS_PATH_DEFAULT,
    DATA_PATH_DEFAULT,
    BATCH_SIZE,
    PRINT_FREQ,
    LEARN_RATE,
    EPOCHS,
    WEIGHTS_PATH_DEFAULT,
    ORIGINAL_DATASET_ROOT_DEFAULT
)

from jigsaw import jigsaw_batch

from training_utils import train_original_dataset, get_model, test_model, load_original_dataset
from utils import generate_results
from rotation_invariance import effective_invariance

parser = argparse.ArgumentParser(description="AutoEval baselines - Rotation Prediction")
parser.add_argument(
    "--model",
    required=True,
    type=str,
    help="the model used to run this script",
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
    "--batch_size",
    required=False,
    type=int,
    default=BATCH_SIZE,
    help="Number of training samples in one batch."
)
parser.add_argument(
    "--dsets",
    required=False,
    default=[],
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
    "--cifar10-original-root",
    required=False,
    default=ORIGINAL_DATASET_ROOT_DEFAULT,
    type=str,
    help="path containing the original CIFAR10 dataset"
)
parser.add_argument(
    "--results-path",
    required=False,
    default=RESULTS_PATH_DEFAULT,
    type=str,
    help="The path to store results."
)
parser.add_argument(
    "--weights-path",
    required=False,
    default=WEIGHTS_PATH_DEFAULT,
    type=str,
    help="The path to store the model weights."
)
parser.add_argument(
    "--recalculate-results",
    action="store_true",
    required=False,
    help="Whether the task should be recalculated over the given dataset paths."
)


def jigsaw_inv_pred(dataloader, model, device, int_to_perm, random=False, exclude_id=True):
    yhat = []
    yhat_t = []
    phat = []
    phat_t = []
    num_permutations = len(int_to_perm)
    for imgs, _ in iter(dataloader):
        imgs.to(device)
        imgs_rot, _ = jigsaw_batch(imgs, num_permutations, int_to_perm, random=random, exclude_id=exclude_id)
        imgs_rot = imgs_rot.to(device)

        with torch.no_grad():
            out_class, _ = model(imgs)
            out_rot, _ = model(imgs_rot)

            out_class_maxes = torch.max(out_class, dim=1)
            out_class_preds = out_class_maxes.indices
            out_class_confs = out_class_maxes.values

            out_jigsaw_maxes = torch.max(out_rot, dim=1)
            out_jigsaw_preds = out_jigsaw_maxes.indices
            out_jigsaw_confs = out_jigsaw_maxes.values

            # repeat each entry of out_class_preds 3 times.
            #out_class_preds = torch.stack([out_class_preds for _ in range(4)], dim=1).flatten()
            out_class_preds = out_class_preds.repeat(1, num_permutations - 1).flatten()
            #out_class_confs = torch.stack([out_class_confs for _ in range(4)], dim=1).flatten()
            out_class_confs = out_class_confs.repeat(1, num_permutations - 1).flatten()

            yhat.append(out_class_preds)
            yhat_t.append(out_jigsaw_preds)

            phat.append(out_class_confs)
            phat_t.append(out_jigsaw_confs)

    yhat = torch.concat(yhat)
    yhat_t = torch.concat(yhat_t)
    phat = torch.concat(phat)
    phat_t = torch.concat(phat_t)
    return effective_invariance(yhat, yhat_t, phat, phat_t)


def main(*ags, **kwargs):
    generate_results(*ags, **kwargs, load_best_fc=False)


if __name__ == "__main__":
    ensure_cwd()
    args = parser.parse_args()
    main(args.dataset,
         args.model,
         "jigsaw_invariance",
         args.data_root,
         args.results_path,
         args.dsets,
         4,
         jigsaw_inv_pred,
         recalculate_results=args.recalculate_results)