import argparse
import os
import sys

sys.path.append(".")

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.resnet import ResNet_SS
from models.repvgg import RepVGG_SS
from utils import CIFAR10NP, TRANSFORM, fit_lr
from training_utils import (
    load_original_cifar_dataset,
    get_model,
    train_ss_fc
)

from eval_utils import (
    eval_train,
    eval_validation
)

valid_models = ["mobilenetv2"] # Temporarily exclude resnet and repvgg to preserve the rotation FC .pt files.

parser = argparse.ArgumentParser(description="AutoEval baselines - Rotation Prediction")
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
    required=False,
    type=bool,
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
    default=50,
    type=float,
    help='Number of epochs for training.'
)
parser.add_argument(
    '--show-graphs',
    default=True,
    type=bool,
    help='True if the graphs of classification accuracy vs jigsaw accuracy should be shown before RMSE calculation'
)

# Assumes that tensor is (nchannels, height, width)
def tensor_rot_90(x):
    return x.flip(2).transpose(1, 2)


def tensor_rot_180(x):
    return x.flip(2).flip(1)


def tensor_rot_270(x):
    return x.transpose(1, 2).flip(2)


def rotate_batch_with_labels(batch, labels):
    images = []
    for img, label in zip(batch, labels):
        if label == 1:
            img = tensor_rot_90(img)
        elif label == 2:
            img = tensor_rot_180(img)
        elif label == 3:
            img = tensor_rot_270(img)
        images.append(img.unsqueeze(0))
    return torch.cat(images)


def rotate_batch(batch, label):
    if label == "rand":
        labels = torch.randint(4, (len(batch),), dtype=torch.long)
    elif label == "expand":
        labels = torch.cat(
            [
                torch.zeros(len(batch), dtype=torch.long),
                torch.zeros(len(batch), dtype=torch.long) + 1,
                torch.zeros(len(batch), dtype=torch.long) + 2,
                torch.zeros(len(batch), dtype=torch.long) + 3,
            ]
        )
        batch = batch.repeat((4, 1, 1, 1))
    else:
        assert isinstance(label, int)
        labels = torch.zeros((len(batch),), dtype=torch.long) + label
    return rotate_batch_with_labels(batch, labels), labels


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


if __name__ == "__main__":
    plt.ion()
    args = parser.parse_args()

    # paths
    dataset_path = args.dataset_path
    model_name = args.model
    grid_length = args.grid_length

    task_name = "rotation"
    train_set = "train_data"
    val_sets = sorted(["cifar10-f-32", "cifar-10.1-c", "cifar-10.1"])
    temp_file_path = f"../temp/{model_name}/{task_name}/"

    # This function must only be run ONCE!!! Computing permutations is O(n!)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get the model given the input parameters.
    model = get_model(model_name, task_name, 4, device, args.train_ss_layer)

    ss_batch_func = lambda inp_batch: rotate_batch(inp_batch, "rand")
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

    ss_predictor_func = lambda dataloader: rotation_pred(dataloader, model, device)
    if args.train_ss_layer or not os.path.exists(f"{temp_file_path}{train_set}.npy"):
        eval_train(dataset_path, temp_file_path, train_set, TRANSFORM, args.batch_size, ss_predictor_func)

    if args.train_ss_layer or not os.path.exists(f"{temp_file_path}val_sets.npy"):
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
        f"===> Linear Regression model for rotation accuracy method with model: {model_name}"
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
           args.show_graphs,
           task_name.capitalize(),
           args.model)
