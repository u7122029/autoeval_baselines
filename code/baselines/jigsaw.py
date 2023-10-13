# Written by Callum Koh (u7122029@anu.edu.au)

import argparse
import sys

import torchvision.transforms.functional as functional

sys.path.append(".")

import numpy as np

import torch
import torch.optim
import torch.utils.data
from pathlib import Path

from utils import (
    construct_permutation_mappings,
    generate_results,
    ensure_cwd,
    VALID_MODELS,
    VALID_DATASETS,
    DATA_PATH_DEFAULT,
    BATCH_SIZE,
    PRINT_FREQ,
    WEIGHTS_PATH_DEFAULT,
    ORIGINAL_DATASET_ROOT_DEFAULT,
    RESULTS_PATH_DEFAULT,
    DEVICE,
    DEFAULT_MAX_JIGSAW_PERMS
)

from training_utils import train_original_dataset

parser = argparse.ArgumentParser(description="AutoEval baselines - Jigsaw Prediction")
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
    "--data-root",
    required=False,
    default=DATA_PATH_DEFAULT,
    type=str,
    help="path containing all datasets (training and validation)"
)
parser.add_argument(
    "--grid_length",
    required=False,
    type=int,
    default=2,
    help="The length of one side of a (square) jigsaw image."
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
    "--train-ss-layer",
    action="store_true",
    required=False,
    default=False,
    help="True if the model's Fully Connected (FC) layer for jigsaw prediction should be trained, and False otherwise."
)
parser.add_argument(
    '--print-freq',
    default=PRINT_FREQ,
    type=int,
    help='Size of intervals between running average loss prints.'
)
parser.add_argument(
    '--lr',
    default=1e-2,
    type=float,
    help='Learning Rate'
)
parser.add_argument(
    '--epochs',
    default=25,
    type=float,
    help='Number of epochs for training.'
)
parser.add_argument(
    '--show-graphs',
    action="store_true",
    default=False,
    help='True if the graphs of classification accuracy vs jigsaw accuracy should be shown after RMSE calculation'
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
    '--show-train-animation',
    action="store_true",
    default=False,
    help='Shows the loss curves during training.'
)
parser.add_argument(
    '--max-permutations',
    required=False,
    default=DEFAULT_MAX_JIGSAW_PERMS,
    type=int,
    help='Dictates the maximum number of jigsaw permutations to use. Should not be larger than (grid_length ** 2)!'
)


# Assumes (n_channels, rows, cols)
def patchify_image(image_tensor, grid_length=2):
    """
    Converts an image into its patch components.
    :param image_tensor: The image as a tensor.
    :param grid_length: The number of rows (and columns) in the partitioning.
    :return: List of patches.
    """
    n_channels, n_rows, n_cols = image_tensor.shape
    row_length = n_rows // grid_length
    col_length = n_cols // grid_length

    patches = []
    rows = torch.split(image_tensor, row_length, 1)
    for row in rows:
        row_patches = torch.split(row, col_length, 2)
        patches += row_patches

    return torch.stack(patches)


def reassemble_jigsaw(patches):
    """
    Reassemble an image given the jigsaw patches.
    :param patches: The jigsaw image patches (not shuffled)
    :return: Image with all the image patches put together in the order they are given.
    """
    grid_length = int(np.sqrt(len(patches)))
    patch_row_length = patches[0].shape[1]
    patch_col_length = patches[0].shape[2]
    reassembled = torch.zeros((3, patch_row_length * grid_length, patch_col_length * grid_length))
    for row in range(grid_length):
        for col in range(grid_length):
            reassembled[:,
            patch_row_length * row:patch_row_length * (row + 1),
            patch_col_length * col:patch_col_length * (col + 1)] = patches[row * grid_length + col]

    return reassembled


def jigsaw_batch_with_labels(batch, labels, int_to_perm):
    images = []
    for img, label in zip(batch, labels):
        perm = int_to_perm[label.item()]["perm"]
        patches = patchify_image(img, args.grid_length)
        permuted = patches[perm, :, :, :]

        # Put image back together
        reassembled = reassemble_jigsaw(permuted)
        images.append(reassembled.unsqueeze(0))
    return torch.cat(images)


def jigsaw_batch(batch, num_permutations, int_to_perm, random=True, exclude_id=False):
    """
    :param batch: The batch
    :param num_permutations: The number of jigsaw permutations
    :param int_to_perm: The permutation index to the actual permutation tuple.
    :param random: True, if the permutation assigned to each image should be random, and False if there should be
    ``num_permutations`` copies of each image associated with each possible permutation.
    :param exclude_id: True, if the identity permutation should be excluded (label index 0).
    :return: The batch of jigsaw labels and the labels themselves.
    """
    if random:
        labels = torch.randint(num_permutations, (len(batch),), dtype=torch.long)
    elif not random and not exclude_id:
        labels = torch.cat(
            [torch.zeros(len(batch), dtype=torch.long) + label_idx for label_idx in range(num_permutations)]
        )
        batch = batch.repeat((num_permutations, 1, 1, 1))
    elif not random and exclude_id:
        labels = torch.cat(
            [torch.zeros(len(batch), dtype=torch.long) + label_idx for label_idx in range(1,num_permutations)]
        )
        batch = batch.repeat((num_permutations - 1, 1, 1, 1))
    else:
        raise Exception(f"Cannot have random={random} and exclude_id={exclude_id} at the same time.")

    return jigsaw_batch_with_labels(batch, labels, int_to_perm), labels


def jigsaw_pred(dataloader, model, device, num_permutations, int_to_perm):
    # return a tuple of (classification accuracy, rotation prediction accuracy)
    outcomes = []
    for imgs, _ in iter(dataloader):
        imgs_jig, labels_jig = jigsaw_batch(imgs, num_permutations, int_to_perm, random=False)
        imgs_jig, labels_jig = imgs_jig.to(device), labels_jig.to(device)
        with torch.no_grad():
            _, out_jig = model(imgs_jig)
            pred_jig = torch.argmax(out_jig, dim=1, keepdim=True)
            outcomes.append(pred_jig.squeeze(1).eq(labels_jig).cpu())
    outcomes = torch.cat(outcomes).numpy()
    return np.mean(outcomes)


def main(model_name,
         data_root,
         cifar10_root,
         dset_paths,
         train_ss_layer,
         batch_size,
         epochs,
         lr,
         print_freq,
         recalculate_results,
         results_path=RESULTS_PATH_DEFAULT,
         device=DEVICE,
         weights_path=WEIGHTS_PATH_DEFAULT,
         show_train_animation=False,
         dataset_name="cifar10",
         grid_length=2,
         max_perms=None):
    data_root = Path(data_root)
    results_path = Path(results_path)
    dset_paths = [Path(i) for i in dset_paths]
    weights_path = Path(weights_path)
    cifar10_root = Path(cifar10_root)

    int_to_perm = construct_permutation_mappings(grid_length,
                                                 num_out_perms=int(max_perms) if max_perms is not None else None)
    num_ss_out = len(int_to_perm)
    task_name = f"jigsaw-grid-len-{grid_length}_max-perm-{num_ss_out}"
    ss_batch_func = lambda inp_batch: jigsaw_batch(inp_batch, num_ss_out, int_to_perm)

    # Get the model given the input parameters.
    best_ss_weights_exists = (weights_path / model_name / dataset_name / task_name / "best.pt").exists()

    # Train the model if required
    if train_ss_layer or not best_ss_weights_exists:
        train_original_dataset(dataset_name,
                               cifar10_root,
                               model_name,
                               num_ss_out,
                               task_name,
                               ss_batch_func,
                               batch_size,
                               epochs,
                               lr,
                               print_freq,
                               weights_path,
                               device=DEVICE,
                               show_train_animation=show_train_animation)

    generate_results(dataset_name,
                     model_name,
                     task_name,
                     data_root,
                     results_path,
                     dset_paths,
                     num_ss_out,
                     lambda dataloader, model, device: jigsaw_pred(dataloader, model, device,
                                                                   num_ss_out, int_to_perm),
                     recalculate_results,
                     device)


if __name__ == "__main__":
    ensure_cwd()
    args = parser.parse_args()
    main(args.model,
         args.data_root,
         args.cifar10_original_root,
         args.dsets,
         args.train_ss_layer,
         args.batch_size,
         args.epochs,
         args.lr,
         args.print_freq,
         args.recalculate_results,
         results_path=args.results_path,
         device=DEVICE,
         weights_path=args.weights_path,
         show_train_animation=args.show_train_animation,
         dataset_name=args.dataset,
         grid_length=args.grid_length,
         max_perms=args.max_permutations)
