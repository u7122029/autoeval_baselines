import argparse
import os
import sys

sys.path.append(".")

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from utils import (
    TRANSFORM,
    fit_lr,
    ensure_cwd,
    VALID_MODELS,
    DEVICE,
    TEMP_PATH_DEFAULT,
    RESULTS_PATH_DEFAULT,
    DEFAULT_DATASET_COND,
    DATA_PATH_DEFAULT,
    BATCH_SIZE,
    PRINT_FREQ,
    LEARN_RATE,
    EPOCHS,
    WEIGHTS_PATH_DEFAULT
)

from training_utils import (
    load_original_cifar_dataset,
    get_model,
    train_ss_fc
)

from utils import (
    generate_results
)

from models.model import Model

parser = argparse.ArgumentParser(description="AutoEval baselines - Rotation Prediction")
parser.add_argument(
    "--model",
    required=True,
    type=str,
    help="the model used to run this script",
    choices=VALID_MODELS
)
parser.add_argument(
    "--dataset-path",
    required=False,
    default=DATA_PATH_DEFAULT,
    type=str,
    help="path containing all datasets (training and validation)",
)
parser.add_argument(
    "--batch_size",
    required=False,
    type=int,
    default=BATCH_SIZE,
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
    "--temp-path",
    required=False,
    default=TEMP_PATH_DEFAULT,
    type=str,
    help="The path to store temporary files."
)
parser.add_argument(
    '--print-freq',
    default=PRINT_FREQ,
    type=int,
    help='Size of intervals between running average loss prints.'
)
parser.add_argument(
    '--lr',
    default=LEARN_RATE,
    type=float,
    help='Learning Rate'
)
parser.add_argument(
    '--epochs',
    default=EPOCHS,
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
    '--use-rand-labels-eval',
    action="store_true",
    default=False,
    help='True if the self-supervision labels during evaluation on the interior and exterior domains should be completely randomised.'
)
parser.add_argument(
    '--reevaluate-domains',
    action="store_true",
    default=False,
    help='True if the model should be reevaluated on the interior and exterior domain datasets.'
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


def rotation_pred(dataloader, model, device, label_method="expand"):
    # return a tuple of (classification accuracy, rotation prediction accuracy)
    correct_rot = []
    for imgs, _ in iter(dataloader):
        imgs_rot, labels_rot = rotate_batch(imgs, label_method)
        imgs_rot, labels_rot = imgs_rot.to(device), labels_rot.to(device)
        with torch.no_grad():
            _, out_rot = model(imgs_rot)
            pred_rot = torch.argmax(out_rot, dim=1, keepdim=True)
            correct_rot.append(pred_rot.squeeze(1).eq(labels_rot).cpu())
    correct_rot = torch.cat(correct_rot).numpy()
    return np.mean(correct_rot)


# label_method = "rand" if use_rand_labels_eval else "expand"
ss_batch_func = lambda inp_batch: rotate_batch(inp_batch, "rand")
ss_predictor_func = lambda dataloader, model, device: rotation_pred(dataloader,
                                                                    model,
                                                                    device)


def train_original_cifar10(data_root: Path,
                           model: Model,
                           task_name: str,
                           batch_func,
                           batch_size: int,
                           epochs: int,
                           lr: float,
                           print_freq: int,
                           weights_path: Path,
                           device=DEVICE,
                           show_train_animation=False):
    if show_train_animation:
        plt.ion()

    train_loader, test_loader = load_original_cifar_dataset(
        data_root, batch_size, device
    )

    train_ss_fc(
        model,
        device,
        train_loader,
        test_loader,
        batch_func,
        task_name,
        epochs,
        lr,
        print_freq=print_freq,
        show_animation=show_train_animation,
        weights_path=weights_path
    )

    if show_train_animation:
        plt.ioff()
        plt.show()


def main(model_name,
         data_root,
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
         show_train_animation=False):

    data_root = Path(data_root)
    results_path = Path(results_path)
    dset_paths = [Path(i) for i in dset_paths]
    weights_path = Path(weights_path)

    task_name = "rotation"

    # Get the model given the input parameters.
    best_ss_weights_exists = (weights_path / model_name / task_name / "best.pt").exists()
    model = get_model(model_name, task_name, 4, device, not train_ss_layer and best_ss_weights_exists)

    # Train the model if required
    if train_ss_layer or not best_ss_weights_exists:
        train_original_cifar10(data_root,
                               model,
                               task_name,
                               ss_batch_func,
                               batch_size,
                               epochs,
                               lr,
                               print_freq,
                               weights_path,
                               device=DEVICE,
                               show_train_animation=show_train_animation)

    generate_results(model_name,
                     task_name,
                     data_root,
                     results_path,
                     dset_paths,
                     4,
                     ss_predictor_func,
                     recalculate_results,
                     device)


if __name__ == "__main__":
    ensure_cwd()
