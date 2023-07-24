import argparse
import os
import sys
sys.path.append(".")

import numpy as np
import torch
import matplotlib.pyplot as plt

from utils import (
    TRANSFORM,
    fit_lr,
    get_dirs,
    VALID_MODELS,
    DEVICE,
    TEMP_PATH_DEFAULT,
    TRAIN_DATA,
    DEFAULT_DATASET_COND,
    JIGSAW_GRID_LENGTH,
    DATA_PATH_DEFAULT, BATCH_SIZE, PRINT_FREQ, LEARN_RATE, EPOCHS, WEIGHTS_PATH_DEFAULT
)

from training_utils import (
    load_original_cifar_dataset,
    get_model,
    train_ss_fc
)

from eval_utils import (
    eval_train,
    eval_validation
)

parser = argparse.ArgumentParser(description="AutoEval baselines - Rotation Prediction")
parser.add_argument(
    "--model",
    required=True,
    type=str,
    help="the model used to run this script",
    choices=VALID_MODELS
)
parser.add_argument(
    "--dataset_path",
    required=False,
    default=DATA_PATH_DEFAULT,
    type=str,
    help="path containing all datasets (training and validation)",
)
parser.add_argument(
    "--grid_length",
    required=False,
    type=int,
    default=JIGSAW_GRID_LENGTH,
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


def main(model_name,
         dataset_path,
         show_graphs,
         train_ss_layer,
         batch_size,
         epochs,
         lr,
         print_freq,
         reevaluate_domains,
         use_rand_labels_eval,
         train_set,
         val_set_paths,
         temp_file_path=TEMP_PATH_DEFAULT,
         device=DEVICE,
         weights_path=WEIGHTS_PATH_DEFAULT,
         show_train_animation=True):
    if show_train_animation:
        plt.ion()

    if dataset_path.endswith('/'):
        dataset_path = dataset_path[:-1]

    task_name = "rotation"

    # Get the model given the input parameters.
    best_ss_weights_exists = os.path.exists(f"{weights_path}/{model_name}/{task_name}/best.pt")
    model = get_model(model_name, task_name, 4, device, not train_ss_layer and best_ss_weights_exists)

    # Train the model if required
    if train_ss_layer or not best_ss_weights_exists:
        ss_batch_func = lambda inp_batch: rotate_batch(inp_batch, "rand")
        train_loader, test_loader = load_original_cifar_dataset(
            device,
            batch_size,
            dataset_path
        )

        train_ss_fc(
            model,
            device,
            train_loader,
            test_loader,
            ss_batch_func,
            task_name,
            epochs,
            lr,
            print_freq=print_freq,
            figure_save_dir=weights_path,
            show_animation=show_train_animation
        )

    # All the below should be after training.
    # The CIFAR-10 datasets used below are non-standard and have been manipulated in various ways.
    # need to do rotation accuracy calculation
    if not os.path.exists(f"{temp_file_path}/{model_name}/{task_name}"):
        os.makedirs(f"{temp_file_path}/{model_name}/{task_name}")

    label_method = "rand" if use_rand_labels_eval else "expand"
    ss_predictor_func = lambda dataloader: rotation_pred(dataloader,
                                                         model,
                                                         device,
                                                         label_method=label_method)

    if reevaluate_domains or train_ss_layer or \
            not os.path.exists(f"{temp_file_path}/{model_name}/{task_name}/{train_set}.npy") or \
            not best_ss_weights_exists:
        eval_train(dataset_path,
                   temp_file_path,
                   train_set,
                   TRANSFORM,
                   batch_size,
                   ss_predictor_func,
                   task_name,
                   model_name)

    if reevaluate_domains or train_ss_layer or \
            not os.path.exists(f"{temp_file_path}/{model_name}/{task_name}/val_sets.npy") or \
            not best_ss_weights_exists:
        eval_validation(
            temp_file_path,
            val_set_paths,
            TRANSFORM,
            batch_size,
            ss_predictor_func,
            task_name,
            model_name
        )

    print(
        f"===> Linear Regression model for rotation accuracy method with model: {model_name}"
    )
    train_x = np.load(f"{temp_file_path}/{model_name}/{task_name}/{train_set}.npy") * 100
    train_y = np.load(f"{temp_file_path}/{model_name}/classification/{train_set}.npy") * 100
    val_x = np.load(f"{temp_file_path}/{model_name}/{task_name}/val_sets.npy") * 100
    val_y = np.load(f"{temp_file_path}/{model_name}/classification/val_sets.npy") * 100
    if show_train_animation:
        plt.ioff()
        plt.show()

    fit_lr(train_x,
           train_y,
           val_x,
           val_y,
           task_name.capitalize(),
           model_name,
           show_graphs=show_graphs,
           save_graphs_dir=f"{temp_file_path}/{model_name}/{task_name}")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.model,
         args.dataset_path,
         args.show_graphs,
         args.train_ss_layer,
         args.batch_size,
         args.epochs,
         args.lr,
         args.print_freq,
         args.reevaluate_domains,
         args.use_rand_labels_eval,
         TRAIN_DATA,
         get_dirs(DATA_PATH_DEFAULT, DEFAULT_DATASET_COND))
