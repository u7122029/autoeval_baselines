import numpy as np
import scipy as sp
import torch
import argparse
import sys
sys.path.append(".")
from utils import (generate_results,
                   VALID_MODELS,
                   VALID_DATASETS,
                   DATA_PATH_DEFAULT,
                   RESULTS_PATH_DEFAULT,
                   DEVICE)

import torch.nn.functional as F

parser = argparse.ArgumentParser(description="AutoEval baselines - Nuclear Norm Baseline")
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
    "--original-root",
    required=False,
    default="C:/ml_datasets",
    type=str,
    help="path containing the original dataset"
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


def nuclear_norm(probs):
    """
    Computes the nuclear norm of a matrix of softmax vectors
    :param prob: N x K sized matrix, where N is the dataset size, and K is the number of output classes.
    :return:
    """
    s = torch.linalg.svdvals(probs)
    return (s.sum() / np.sqrt(min(probs.shape) * probs.shape[0])).item()


def nuclear_norm_pred(dataloader, model, device):
    """
    Predictor function for the nuclear norm baseline.
    :param dataloader: The dataloader (only has one batch that is the size of the entire dataset)
    :param model: the model
    :param device: the device
    :return: the nuclear norm for the dataloader.
    """
    batches = iter(dataloader)
    batch, labels = next(batches)
    batch = batch.to(device)

    class_preds, _ = model(batch)
    class_probs = F.softmax(class_preds, dim=1)
    return nuclear_norm(class_probs)


def main(dataset_name, model_name, data_root, results_path, dset_paths, **kwargs):
    generate_results(dataset_name,
                     model_name,
                     "nuclear_norm",
                     data_root,
                     results_path,
                     dset_paths,
                     10, # This doesn't matter.
                     nuclear_norm_pred,
                     load_best_fc=False,
                     **kwargs)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.dataset, args.model, args.data_root, args.results_path, args.dsets,
         recalculate_results=args.recalculate_results,
         device=DEVICE)