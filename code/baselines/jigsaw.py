# Written by Callum Koh (u7122029@anu.edu.au)

import argparse
import os
import sys
import random
import torchvision.transforms.functional as functional

sys.path.append(".")

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import torch
from tqdm import tqdm

from models.resnet_jigsaw import ResNetJigsaw
from utils import CIFAR10NP, TRANSFORM, construct_permutation_mappings


parser = argparse.ArgumentParser(description="AutoEval baselines - Jigsaw Prediction")
parser.add_argument(
    "--model",
    required=True,
    type=str,
    help="the model used to run this script"
)
parser.add_argument(
    "--dataset_path",
    required=True,
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
    "--train_jigsaw_fc",
    required=False,
    type=bool,
    default=False,
    help="True if the model's Fully Connected (FC) layer for jigsaw prediction should be trained, and False otherwise."
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

    for row in range(grid_length):
        for col in range(grid_length):
            patches.append((grid_length * row + col,
                            functional.crop(image_tensor, row*row_length, col*col_length, row_length, col_length))
                            )

    return patches

def reassemble_jigsaw(patches):
    """
    Reassemble an image given the jigsaw patches.
    :param patches: The jigsaw image patches (shuffled)
    :return: Image with all the image patches put together in the order they are given.
    """
    grid_length = int(np.sqrt(len(patches)))
    patch_row_length = patches[0].shape[1]
    patch_col_length = patches[0].shape[2]
    reassembled = torch.zeros((3, patch_row_length * grid_length, patch_col_length * grid_length))
    for row in range(grid_length):
        for col in range(grid_length):
            reassembled[:,
                        patch_row_length * row:patch_row_length * (row+1),
                        patch_col_length * col:patch_col_length * (col+1)] = patches[row * grid_length + col]

    return reassembled

def image_to_jigsaw(image_tensor, grid_length=2):
    """
    Transforms an image into a jigsaw image.
    :param image_tensor: The image tensor
    :param grid_length: The length (and width) of one jigsaw piece.
    :return:
    """
    patches = patchify_image(image_tensor, grid_length)
    random.shuffle(patches)
    permutation = tuple([i for i,_ in patches])
    permuted_patches = [j for _,j in patches]
    stitched_jigsaw = reassemble_jigsaw(permuted_patches)
    return stitched_jigsaw, permutation

def image_batch_to_jigsaw_batch(batch):
    jigsaws = []
    labels = []
    for img in batch:
        out = image_to_jigsaw(img)
        jigsaws.append(out[0])
        labels.append(out[1])
    return jigsaws, labels

def rotation_pred(dataloader, model, device):
    # return a tuple of (classification accuracy, rotation prediction accuracy)
    correct_rot = []
    for imgs, _ in iter(dataloader):
        imgs_rot, labels_rot = rotate_batch(imgs, "expand")
        imgs_rot, labels_rot = imgs_rot.to(device), labels_rot.to(device)
        with torch.no_grad():
            _, out_rot = model(imgs_rot)
            pred_rot = torch.argmax(out_rot, dim=1, keepdim=True)
            correct_rot.append(pred_rot.squeeze(1).eq(labels_rot).cpu())
    correct_rot = torch.cat(correct_rot).numpy()
    return np.mean(correct_rot)

def train_model_FC(model, device, dataloader):
    # Freeze all layers, then unfreeze jigsaw FC layer
    for param in model.parameters():
        param.requires_grad = False
    model.fc_jigsaw.weight.requires_grad = True
    model.fc_jigsaw.bias.requires_grad = True

    epochs = 20
    learning_rate = 1e-2
    momentum = 0.9
    weight_decay = 1e-4
    print_freq = 10
    layers = 110

def get_model(name, grid_length, device):
    if name == "resnet":
        model = ResNetJigsaw(grid_length)
        model_state = model.state_dict()
        fc_jig_weights = torch.load(
            "../model_weights/resnet-jig-fc.pt", map_location=torch.device("cpu")
        )
    else:
        raise NameError(f"Model name {name} does not exist.") # TODO: get list of model names automatically.

    # load the rotation FC layer weights
    for key, value in fc_jig_weights.items():
        model_state[key] = value
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    return model

if __name__ == "__main__":
    # paths
    args = parser.parse_args()
    dataset_path = args.dataset_path
    model_name = args.model
    grid_length = args.grid_length
    perm_to_int, int_to_perm = construct_permutation_mappings(grid_length)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get the model given the input parameters.
    model = get_model(model_name, grid_length, device)

    train_set = "train_data"
    val_sets = sorted(["cifar10-f-32", "cifar-10.1-c", "cifar-10.1"])
    temp_file_path = f"../temp/{model_name}/jigsaw/"





