# Written by Callum Koh (u7122029@anu.edu.au)

import argparse
import os
import sys

import torchvision.transforms.functional as functional

sys.path.append(".")

import numpy as np

import torch
import torch.optim
import torch.utils.data

import matplotlib.pyplot as plt  # visualisation

from utils import (
    TRANSFORM,
    construct_permutation_mappings,
    fit_lr,
    valid_models
)

from training_utils import (
    load_original_cifar_dataset,
    get_model,
    train_ss_fc
)

from eval_utils import (
    eval_validation,
    eval_train
)

parser = argparse.ArgumentParser(description="AutoEval baselines - Jigsaw Prediction")
parser.add_argument(
    "--model",
    required=True,
    type=str,
    help="the model used to run this script",
    choices=valid_models
)
parser.add_argument(
    "--dataset_path",
    required=False,
    default="data",
    type=str,
    help="path containing all datasets (training and validation)",
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
    default=64,
    help="Number of training samples in one batch."
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
    default=100,
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
    '--reevaluate-domains',
    action="store_true",
    default=False,
    help='True if the model should be reevaluated on the interior and exterior domain datasets.'
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
    positions = []

    for row in range(grid_length):
        for col in range(grid_length):
            positions.append(grid_length * row + col)
            patches.append(functional.crop(image_tensor, row * row_length, col * col_length, row_length, col_length))

    return torch.stack(patches), positions


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
        patches, positions = patchify_image(img, args.grid_length)
        permuted = patches[perm, :, :, :]

        # Put image back together
        reassembled = reassemble_jigsaw(permuted)
        images.append(reassembled.unsqueeze(0))
    return torch.cat(images)


def jigsaw_batch(batch, num_permutations, int_to_perm, random=True):
    """
    :param batch: The batch
    :param num_permutations: The number of jigsaw permutations
    :param int_to_perm: The permutation index to the actual permutation tuple.
    :param random: True, if the permutation assigned to each image should be random, and False if there should be
    ``num_permutations`` copies of each image associated with each possible permutation.
    :return: The batch of jigsaw labels and the labels themselves.
    """
    if random:
        labels = torch.randint(num_permutations, (len(batch),), dtype=torch.long)
    else:
        labels = torch.cat(
            [torch.zeros(len(batch), dtype=torch.long) + label_idx for label_idx in range(num_permutations)]
        )
        batch = batch.repeat((num_permutations, 1, 1, 1))
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


if __name__ == "__main__":
    plt.ion()
    args = parser.parse_args()

    # paths
    dataset_path = args.dataset_path
    model_name = args.model
    grid_length = args.grid_length

    task_name = "jigsaw"
    train_set = "train_data"
    val_sets = sorted(["cifar10-f-32", "cifar-10.1-c", "cifar-10.1"])
    temp_file_path = f"../temp/{model_name}/jigsaw/"

    # This function must only be run ONCE!!! Computing permutations is O(n!)
    int_to_perm = construct_permutation_mappings(grid_length)
    num_permutations = len(int_to_perm)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get the model given the input parameters.
    model = get_model(model_name, task_name, num_permutations, device, not args.train_ss_layer)

    ss_batch_func = lambda inp_batch: jigsaw_batch(inp_batch, model.num_ss_classes, int_to_perm)
    # Train the model if required
    if args.train_ss_layer:
        train_loader, test_loader = load_original_cifar_dataset(
            device,
            args.batch_size,
            args.dataset_path
        )

        train_ss_fc(
            model,
            device,
            train_loader,
            test_loader,
            ss_batch_func,
            task_name,
            args.epochs,
            args.lr,
            print_freq=args.print_freq
        )

    # All the below should be after training.
    # The CIFAR-10 datasets used below are non-standard and have been manipulated in various ways.
    # need to do rotation accuracy calculation
    if not os.path.exists(temp_file_path):
        os.makedirs(temp_file_path)

    ss_predictor_func = lambda dataloader: jigsaw_pred(dataloader, model, device, num_permutations, int_to_perm)
    if args.train_ss_layer or args.reevaluate_domains or not os.path.exists(f"{temp_file_path}{train_set}.npy"):
        eval_train(dataset_path, temp_file_path, train_set, TRANSFORM, args.batch_size, ss_predictor_func)

    if args.train_ss_layer or args.reevaluate_domains or not os.path.exists(f"{temp_file_path}val_sets.npy"):
        eval_validation(
            dataset_path,
            temp_file_path,
            val_sets,
            TRANSFORM,
            args.batch_size,
            ss_predictor_func
        )

    # if the calculation of rotation accuracy is finished
    # calculate the linear regression model (accuracy in %)
    print(
        f"===> Linear Regression model for jigsaw accuracy method with model: {model_name}"
    )
    train_x = np.load(f"{temp_file_path}{train_set}.npy") * 100
    train_y = np.load(f"../temp/{model_name}/acc/{train_set}.npy") * 100
    val_x = np.load(f"{temp_file_path}val_sets.npy") * 100
    val_y = np.load(f"../temp/{model_name}/acc/val_sets.npy") * 100
    plt.ioff()
    plt.show()
    fit_lr(train_x,
           train_y,
           val_x,
           val_y,
           task_name.capitalize(),
           args.model,
           show_graphs=args.show_graphs)
