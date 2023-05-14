import time

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from models.alexnet import AlexNet_SS
from models.densenet import DenseNet_SS
from models.inceptionv3 import Inceptionv3_SS
from models.lenet5 import LeNet5_SS
from models.linear import Linear_SS
from models.mobilenetv2 import MobileNet_SS
from models.obc import OBC_SS
from models.repvgg import RepVGG_SS
from models.resnet import ResNet_SS
from models.shufflenet import ShuffleNet_SS
from utils import (
    AverageMeter,
    adjust_learning_rate,
    save_checkpoint
)


def load_original_cifar_dataset(device, batch_size, dataset_path):
    """
    Load the original cifar-10 datasets.
    :param device: The device that the datasets should be on.
    :param batch_size: The size of each batch
    :param dataset_path: The path of the dataset
    :return: The training and testing datasets.
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
        datasets.CIFAR10(dataset_path, train=True, download=True, transform=transform_train),
        batch_size=batch_size, shuffle=True, **kwargs
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(dataset_path, train=False, transform=transform_test),
        batch_size=batch_size, shuffle=False, **kwargs
    )

    return train_loader, test_loader


def get_model(name, task, num_ss_classes, device, load_best_fc=True):
    """
    :param name: The name of the backbone model
    :param task: The self-supervision task
    :param num_ss_classes: The number of classes in the self-supervision task
    :param device: The device the model should run on
    :param train_ss_fc: Whether to train the self-supervision FC layer (True) or not (False).
    :return: Instance of the model, with backbone model weights preloaded.
    """

    if "resnet" in name:
        version = int(name.replace("resnet", ""))
        model = ResNet_SS(version, num_ss_classes)
    elif name == "repvgg":
        model = RepVGG_SS(num_ss_classes)
    elif name == "mobilenetv2":
        model = MobileNet_SS(num_ss_classes)
    elif "densenet" in name:
        version = int(name.replace("densenet", ""))
        model = DenseNet_SS(version, num_ss_classes)
    elif name == "shufflenet":
        model = ShuffleNet_SS(num_ss_classes)
    elif name == "inception_v3":
        model = Inceptionv3_SS(num_ss_classes)
    elif name == "linear":
        model = Linear_SS(num_ss_classes)
    elif name == "alexnet":
        model = AlexNet_SS(num_ss_classes)
    elif name == "lenet5":
        model = LeNet5_SS(num_ss_classes)
    elif name == "obc":
        model = OBC_SS(num_ss_classes, device)
    else:
        # Absolutely impossible case since this is covered by argparse.
        # If this NameError occurs please check the choices in the arg parser.
        raise NameError(f"Model name {name} does not exist.")

    if load_best_fc:
        # If we are not training the self-supervision FC layer, we should try to load in its best checkpoint
        model.load_ss_fc(f"../model_weights/{name}-{task}-fc.pt", is_local=True)
        model.eval()

    model.to(device)
    return model


def train_epoch(train_loader, ss_batch_func, model, device, criterion, optimizer, epoch, print_freq):
    """
    :param train_loader: The training dataloader.
    :param ss_batch_func: Function to transform each batch from the training dataloader to suit the self-supervision task.
    :param model: The model to train.
    :param device: The device to run the datasets and model on.
    :param criterion: The criterion.
    :param optimizer: The optimiser.
    :param epoch: The epoch number.
    :param print_freq: The frequency at which the losses should be printed.
    :return:
    """

    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    # top1 = AverageMeter()

    # As backbone is freezed, change it to evaluation mode
    model.eval()

    for i, (inp, _) in enumerate(train_loader):
        end = time.time()
        # target = target.to(device)
        inp = inp.to(device)

        # Classification backbone frozen. Will always give 94.37% accuracy.
        # output, _ = model(inp)
        # loss_cls = criterion(output, target)

        # self-supervised
        inputs_ssh, labels_ssh = ss_batch_func(inp)
        inputs_ssh, labels_ssh = inputs_ssh.to(device), labels_ssh.to(device)
        _, outputs_ssh = model(inputs_ssh)
        loss_ssh = criterion(outputs_ssh, labels_ssh)

        # simply add two tasks' losses
        # loss = loss_cls + loss_ssh

        # measure accuracy and record loss
        # prec1 = accuracy(output, target, topk=(1,))[0] # prec1 will always be the same.
        losses.update(loss_ssh.item(), inp.size(0))
        # top1.update(prec1.item(), inp.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_ssh.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)

        if i % print_freq == 0:
            print(f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t"
                  f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                  # f"Loss_Total {losses.val:.4f} ({losses.avg:.4f})\t"
                  f"Loss_SSH {loss_ssh:.4f} (Running avg: {losses.avg:.4f})\t"
                  # f"Prec@1 {top1.val:.3f} ({top1.avg:.3f})"
                  )
    return losses


def test_model(test_dataloader, model, device, ss_batch_func=None):
    """
    :param test_dataloader: The test dataloader
    :param model: The model to run the test dataloader over
    :param device: The device
    :param ss_batch_func: The function to convert each batch from test_dataloader to suit the self-supervision task.
                            If None, then we test for classification accuracy.
    :return: The accuracy of the model on the test dataloader and the corresponding losses for each batch.
    """
    criterion = nn.CrossEntropyLoss()
    model.eval()
    correct = []
    losses = AverageMeter()

    for batch_idx, (inputs, labels) in enumerate(test_dataloader):
        if ss_batch_func:
            inputs, labels = ss_batch_func(inputs)

        inputs, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            class_out, ss_out = model(inputs)
            loss = criterion(ss_out if ss_batch_func else class_out, labels)
            losses.update(loss.item(), inputs.size(0))

            _, predicted = ss_out.max(1) if ss_batch_func else class_out.max(1)
            correct.append(predicted.eq(labels).cpu())

    acc = torch.cat(correct).numpy().mean() * 100.0

    print(f"{'self-supervised' if ss_batch_func else 'classification'} average: {acc:.4f}%")

    return acc, losses


def train_ss_fc(
        model,
        device,
        train_loader,
        test_loader,
        ss_batch_func,
        ss_task_name,
        epochs=50,
        learning_rate=1e-2,
        momentum=0.9,
        weight_decay=1e-4,
        print_freq=100):
    # Set up figure for plotting losses per epoch
    figure = plt.figure()
    ax = figure.add_subplot(111)
    ax.set_title(f"Loss for each Epoch ({model.model_name})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    train_loss_plot, = ax.plot([], [], 'ro-', label="Train Loss")
    val_loss_plot, = ax.plot([], [], "bx-", label="Val Loss")
    ax.legend()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                learning_rate,
                                momentum=momentum,
                                nesterov=True,
                                weight_decay=weight_decay)

    # best_class_acc = 0
    best_ss_acc = 0

    # Training loop.
    for epoch in range(epochs):
        adjust_learning_rate(optimizer, epoch, learning_rate)

        # train for one epoch
        train_losses = train_epoch(
            train_loader,
            ss_batch_func,
            model,
            device,
            criterion,
            optimizer,
            epoch,
            print_freq
        )

        # evaluate on test set for image classification.
        # Class accuracy stays constant at 94.37%. So we don't need to calculate it.
        # class_acc = test_class(test_loader, model, device, args)

        # evaluate on test set for self-supervised prediction
        ss_acc, val_losses = test_model(
            test_loader,
            model,
            device,
            ss_batch_func
        )

        # Update loss graph
        train_loss_plot.set_xdata(np.append(train_loss_plot.get_xdata(), epoch))
        train_loss_plot.set_ydata(np.append(train_loss_plot.get_ydata(), train_losses.avg))

        val_loss_plot.set_xdata(np.append(val_loss_plot.get_xdata(), epoch))
        val_loss_plot.set_ydata(np.append(val_loss_plot.get_ydata(), val_losses.avg))

        ax.relim()
        ax.autoscale_view(True, True, True)
        figure.canvas.draw()
        figure.canvas.flush_events()

        # remember best prec@1 and save checkpoint
        is_best = ss_acc > best_ss_acc
        # best_class_acc = max(class_acc, best_class_acc)
        best_ss_acc = max(ss_acc, best_ss_acc)
        save_checkpoint(
            model.state_dict(),  # We don't need the epoch number of the best ss_acc
            is_best,
            model.model_name,
            ss_task_name
        )
    print('Best self-supervised accuracy (test set):', best_ss_acc)
