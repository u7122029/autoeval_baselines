import argparse
import sys
sys.path.append(".")

import torch

from utils import (
    ensure_cwd,
    VALID_MODELS,
    VALID_DATASETS,
    RESULTS_PATH_DEFAULT,
    DATA_PATH_DEFAULT,
    BATCH_SIZE,
    WEIGHTS_PATH_DEFAULT,
    ORIGINAL_DATASET_ROOT_DEFAULT
)

from rotation import rotate_batch
from utils import generate_results

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


def effective_invariance(yhat: torch.Tensor, yhat_t: torch.Tensor, phat, phat_t):
    """
    Computes the average effective invariance for each vector.
    :param yhat:
    :param yhat_t:
    :param phat:
    :param phat_t:
    :return:
    """
    return torch.mean((yhat == yhat_t).float() * torch.sqrt(phat * phat_t)).item()


def rotation_inv_pred(dataloader, model, device, label_method="expand_exclude_id"):
    yhat = []
    yhat_t = []
    phat = []
    phat_t = []
    for imgs, _ in iter(dataloader):
        imgs = imgs.to(device)

        imgs_rot, _ = rotate_batch(imgs, label_method) # we don't care about the rotation labels.
        imgs_rot = imgs_rot.to(device)

        with torch.no_grad():
            out_class, _ = model(imgs)
            out_rot, _ = model(imgs_rot)
            if label_method == "expand_exclude_id":
                # repeat each entry of out_class_preds 3 times.
                out_class = out_class.repeat(3,1)

            out_class_maxes = torch.max(torch.softmax(out_class, dim=1), dim=1)
            out_class_preds = out_class_maxes.indices
            out_class_confs = out_class_maxes.values

            out_rot_maxes = torch.max(torch.softmax(out_rot, dim=1), dim=1)
            out_rot_preds = out_rot_maxes.indices
            out_rot_confs = out_rot_maxes.values

            yhat.append(out_class_preds)
            yhat_t.append(out_rot_preds)

            phat.append(out_class_confs)
            phat_t.append(out_rot_confs)

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
         "rotation_invariance",
         args.data_root,
         args.results_path,
         args.dsets,
         4,
         rotation_inv_pred,
         recalculate_results=args.recalculate_results)