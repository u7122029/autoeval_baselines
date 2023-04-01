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
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt # visualisation

from models.resnet_jigsaw import ResNetJigsaw
from utils import CIFAR10NP, TRANSFORM, construct_permutation_mappings, AverageMeter, \
    save_checkpoint, adjust_learning_rate, accuracy

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
    "--batch_size",
    required=False,
    type=int,
    default=64,
    help="Number of training samples in one batch."
)
parser.add_argument(
    "--train_jigsaw_fc",
    required=False,
    type=bool,
    default=False,
    help="True if the model's Fully Connected (FC) layer for jigsaw prediction should be trained, and False otherwise."
)
parser.add_argument(
    '--print-freq',
    default=50,
    type=int,
    help='print frequency (default: 10)'
)
parser.add_argument(
    '--lr',
    default=1e-2,
    type=int,
    help='Learning Rate'
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
            patches.append(functional.crop(image_tensor, row*row_length, col*col_length, row_length, col_length))

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
                        patch_row_length * row:patch_row_length * (row+1),
                        patch_col_length * col:patch_col_length * (col+1)] = patches[row * grid_length + col]

    return reassembled

def jigsaw_batch_with_labels(batch, labels, int_to_perm):
    images = []
    for img, label in zip(batch, labels):
        perm = int_to_perm[label.item()]["perm"]
        patches, positions = patchify_image(img, args.grid_length)
        permuted = patches[perm,:,:,:]

        # Put image back together
        reassembled = reassemble_jigsaw(permuted)
        images.append(reassembled.unsqueeze(0))
    return torch.cat(images)

def jigsaw_batch(batch, num_permutations, int_to_perm):
    labels = torch.randint(num_permutations, (len(batch),), dtype=torch.long)
    return jigsaw_batch_with_labels(batch, labels, int_to_perm), labels

def jigsaw_pred(dataloader, model, device, num_permutations, int_to_perm):
    # return a tuple of (classification accuracy, rotation prediction accuracy)
    outcomes = []
    for imgs, _ in iter(dataloader):
        imgs_jig, labels_jig = jigsaw_batch(imgs, num_permutations, int_to_perm)
        imgs_jig, labels_jig = imgs_jig.to(device), labels_jig.to(device)
        with torch.no_grad():
            _, out_jig = model(imgs_jig)
            pred_jig = torch.argmax(out_jig, dim=1, keepdim=True)
            outcomes.append(pred_jig.squeeze(1).eq(labels_jig).cpu())
    outcomes = torch.cat(outcomes).numpy()
    return np.mean(outcomes)


def load_dataset(args, device):
    """
    Loads the regular CIFAR-10 dataset from the PyTorch dataset repository.
    :param args: The arguments passed into the program.
    :param device: The device to be used. (cuda or cpu)
    :return: The CIFAR-10 training and testing dataloaders.
    """

    normalize = transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    )

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            'data', train=True, download=True, transform=transform_train),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=False, transform=transform_test),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader


def test_class(val_loader, model, device, args):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    criterion = nn.CrossEntropyLoss()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (inp, target) in enumerate(val_loader):
        target = target.to(device)
        inp = inp.to(device)

        # compute output
        with torch.no_grad():
            output, _ = model(inp)
            loss = criterion(output, target)
            # measure accuracy and record loss
            prec1 = accuracy(output, target, topk=(1,))[0]
            losses.update(loss.item(), inp.size(0))
            top1.update(prec1.item(), inp.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # No need to print losses for classification since the backbone is frozen.
        # Printing is slow.
        """if i % args.print_freq == 0:
            print(
                'Test: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1)
            )"""

    print('Classification Accuracy (top1): {top1.avg:.4f}%'.format(top1=top1))
    return top1.avg


def train_epoch(train_loader, model, device, criterion, optimizer, epoch, args, num_permutations, int_to_perm):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # As backbone is freezed, change it to evaluation mode
    model.eval()

    for i, (inp, target) in enumerate(train_loader):
        end = time.time()
        target = target.to(device)
        inp = inp.to(device)

        # compute output
        output, _ = model(inp)
        loss_cls = criterion(output, target)

        # self-supervised
        inputs_ssh, labels_ssh = jigsaw_batch(inp, num_permutations, int_to_perm)
        inputs_ssh, labels_ssh = inputs_ssh.to(device), labels_ssh.to(device)
        _, outputs_ssh = model(inputs_ssh)
        loss_ssh = criterion(outputs_ssh, labels_ssh)

        # simply add two tasks' losses
        '''
        The users could also choose to only use semantic classification loss to train the backbone
        '''
        loss = loss_cls + loss_ssh

        # measure accuracy and record loss
        prec1 = accuracy(output, target, topk=(1,))[0]
        losses.update(loss.item(), inp.size(0))
        top1.update(prec1.item(), inp.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)

        if i % args.print_freq == 0:
            print(f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t"
                  f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                  f"Loss_Total {losses.val:.4f} ({losses.avg:.4f})\t"
                  f"Loss_SSH {loss_ssh:.4f}\t"
                  f"Prec@1 {top1.val:.3f} ({top1.avg:.3f})"
            )

def test_jigsaw(dataloader, model, device, num_permutations, int_to_perm):
    criterion = nn.CrossEntropyLoss(reduction='none')
    model.eval()
    correct = []
    losses = []
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = jigsaw_batch(inputs, num_permutations, int_to_perm)
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            _, outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.cpu())
            _, predicted = outputs.max(1)
            correct.append(predicted.eq(labels).cpu())
    acc = torch.cat(correct).numpy().mean() * 100
    print('Jigsaw self-supervised.avg: {:.4f}%'.format(acc))

    return acc


def train_jigsaw_FC(model, device, train_loader, test_loader, num_permutations, int_to_perm):
    epochs = 25
    learning_rate = 8e-3
    momentum = 0.9
    weight_decay = 1e-4

    # Freeze all layers, then unfreeze jigsaw FC layer
    for param in model.parameters():
        param.requires_grad = False
    model.fc_jigsaw.weight.requires_grad = True
    model.fc_jigsaw1.weight.requires_grad = True
    model.fc_jigsaw2.weight.requires_grad = True
    model.fc_jigsaw.bias.requires_grad = True
    model.fc_jigsaw1.bias.requires_grad = True
    model.fc_jigsaw2.bias.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                learning_rate,
                                momentum=momentum,
                                nesterov=True,
                                weight_decay=weight_decay)

    best_class_acc = 0
    best_rot_acc = 0

    # Training loop.
    for epoch in range(epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_epoch(train_loader, model, device, criterion, optimizer, epoch, args, num_permutations, int_to_perm)

        # evaluate on test set for image classification.
        class_acc = test_class(test_loader, model, device, args)

        # evaluate on test set for rotation prediction
        rot_acc = test_jigsaw(test_loader, model, device, num_permutations, int_to_perm)

        # remember best prec@1 and save checkpoint
        is_best = class_acc > best_class_acc
        best_class_acc = max(class_acc, best_class_acc)
        best_rot_acc = max(rot_acc, best_rot_acc)
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_class_acc,
            },
            is_best,
            "resnet",
            "jigsaw"
        )
    print('Best classification accuracy:', best_class_acc)
    print('Best rotation accuracy:', best_rot_acc)


def get_model(name, num_permutations, device):
    fc_jig_weights = None
    if name == "resnet":
        model = ResNetJigsaw(num_permutations)
        model_state = model.state_dict()
        if not args.train_jigsaw_fc:
            fc_jig_weights = torch.load("../model_weights/resnet-jig-fc.pt", map_location=torch.device("cpu"))
    else:
        raise NameError(f"Model name {name} does not exist.") # TODO: get list of model names automatically.

    if fc_jig_weights and not args.train_jigsaw_fc:
        # load the rotation FC layer weights
        for key, value in fc_jig_weights.items():
            model_state[key] = value
        model.load_state_dict(model_state)
        model.eval()

    model.to(device)
    return model

if __name__ == "__main__":
    args = parser.parse_args()

    # paths
    dataset_path = args.dataset_path
    model_name = args.model
    grid_length = args.grid_length

    train_set = "train_data"
    val_sets = sorted(["cifar10-f-32", "cifar-10.1-c", "cifar-10.1"])
    temp_file_path = f"../temp/{model_name}/jigsaw/"

    # This function must only be run ONCE!!! Computing permutations is O(n!)
    int_to_perm = construct_permutation_mappings(grid_length)
    num_permutations = len(int_to_perm)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, test_loader = load_dataset(args, device)

    # Get the model given the input parameters.
    model = get_model(model_name, num_permutations, device)

    # Train the model if required
    if args.train_jigsaw_fc:
        train_jigsaw_FC(model, device, train_loader,test_loader,num_permutations,int_to_perm)

    # All the below should be after training.
    # The CIFAR-10 datasets used below are non-standard and have been manipulated in various ways.
    # need to do rotation accuracy calculation
    if not os.path.exists(temp_file_path):
        os.makedirs(temp_file_path)

    if args.train_jigsaw_fc or not os.path.exists(f"{temp_file_path}{train_set}.npy"):
        # Load the training set data.
        train_path = f"{dataset_path}/{train_set}"
        train_candidates = []
        for file in sorted(os.listdir(train_path)):
            if file.endswith(".npy") and file.startswith("new_data"):
                train_candidates.append(file)

        print(f"===> Calculating rotation accuracy for {train_set}")
        jigsaw_acc = np.zeros(len(train_candidates))
        for i, candidate in enumerate(tqdm(train_candidates)):
            data_path = f"{train_path}/{candidate}"
            label_path = f"{train_path}/labels.npy"

            train_dataloader = torch.utils.data.DataLoader(
                dataset=CIFAR10NP(
                    data_path=data_path,
                    label_path=label_path,
                    transform=TRANSFORM,
                ),
                batch_size=args.batch_size,
                shuffle=False,
            )
            jigsaw_acc[i] = jigsaw_pred(train_dataloader, model, device, num_permutations, int_to_perm)

        np.save(f"{temp_file_path}{train_set}.npy", jigsaw_acc)

    if args.train_jigsaw_fc or not os.path.exists(f"{temp_file_path}val_sets.npy"):
        # load validation set.
        val_candidates = []
        val_paths = [f"{dataset_path}/{set_name}" for set_name in val_sets]
        for val_path in val_paths:
            for file in sorted(os.listdir(val_path)):
                val_candidates.append(f"{val_path}/{file}")

        jigsaw_acc = np.zeros(len(val_candidates))
        print(f"===> Calculating rotation accuracy for validation sets")

        for i, candidate in enumerate(tqdm(val_candidates)):
            data_path = f"{candidate}/data.npy"
            label_path = f"{candidate}/labels.npy"

            dataloader = torch.utils.data.DataLoader(
                dataset=CIFAR10NP(
                    data_path=data_path,
                    label_path=label_path,
                    transform=TRANSFORM,
                ),
                batch_size=args.batch_size,
                shuffle=False,
            )
            jigsaw_acc[i] = jigsaw_pred(dataloader, model, device, num_permutations, int_to_perm)

        np.save(f"{temp_file_path}val_sets.npy", jigsaw_acc)

    # if the calculation of rotation accuracy is finished
    # calculate the linear regression model (accuracy in %)
    print(
        f"===> Linear Regression model for rotation accuracy method with model: {model_name}"
    )
    train_x = np.load(f"{temp_file_path}{train_set}.npy") * 100
    train_y = np.load(f"../temp/{model_name}/acc/{train_set}.npy") * 100
    val_x = np.load(f"{temp_file_path}val_sets.npy") * 100
    val_y = np.load(f"../temp/{model_name}/acc/val_sets.npy") * 100

    lr = LinearRegression()
    lr.fit(train_x.reshape(-1, 1), train_y)
    # predictions will have 6 decimals
    val_y_hat = np.round(lr.predict(val_x.reshape(-1, 1)), decimals=6)
    rmse_loss = mean_squared_error(y_true=val_y, y_pred=val_y_hat, squared=False)
    print(f"The RMSE on validation set is: {rmse_loss}")





