import argparse
import sys
sys.path.append(".")

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

from training_utils import train_original_dataset
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
    '--show-train-animation',
    action="store_true",
    default=False,
    help='Shows the loss curves during training.'
)
parser.add_argument(
    '--use-rand-labels-eval',
    action="store_true",
    default=False,
    help='True if the self-supervision labels during evaluation on the interior and exterior domains should be '
         'completely randomised.'
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


# Assumes that tensor is (n_channels, height, width)
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
    elif label == "expand_exclude_id":
        labels = torch.cat(
            [
                torch.zeros(len(batch), dtype=torch.long) + 1,
                torch.zeros(len(batch), dtype=torch.long) + 2,
                torch.zeros(len(batch), dtype=torch.long) + 3,
            ]
        )
        batch = batch.repeat((3, 1, 1, 1))
    else:
        assert isinstance(label, int)
        labels = torch.zeros((len(batch),), dtype=torch.long) + label
    return rotate_batch_with_labels(batch, labels), labels


def rotation_pred(dataloader, model, device, label_method="expand"):
    correct_rot = []
    for imgs, _ in iter(dataloader):
        imgs_rot, labels_rot = rotate_batch(imgs, label_method)
        imgs_rot, labels_rot = imgs_rot.to(device), labels_rot.to(device)
        with torch.no_grad():
            _, out_rot = model(imgs_rot)
            pred_rot = torch.argmax(out_rot, dim=1)
            correct_rot.append(pred_rot.eq(labels_rot).float())
    correct_rot = torch.cat(correct_rot)
    return torch.mean(correct_rot).item()


# label_method = "rand" if use_rand_labels_eval else "expand"
ss_batch_func = lambda inp_batch: rotate_batch(inp_batch, "rand")


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
         dataset_name="cifar10"):
    data_root = Path(data_root)
    results_path = Path(results_path)
    dset_paths = [Path(i) for i in dset_paths]
    weights_path = Path(weights_path)
    cifar10_root = Path(cifar10_root)

    task_name = "rotation"

    # Get the model given the input parameters.
    best_ss_weights_exists = (weights_path / model_name / dataset_name / task_name / "best.pt").exists()

    # Train the model if required
    if train_ss_layer or not best_ss_weights_exists:
        train_original_dataset(dataset_name,
                               cifar10_root,
                               model_name,
                               4,
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
                     4,
                     rotation_pred,
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
         dataset_name=args.dataset)
