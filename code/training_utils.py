import time
import shutil
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional
import torch.utils.data
from torch.utils.data import ConcatDataset
import torchvision.datasets as datasets

from utils import (
    DEVICE,
    EPOCHS,
    MOMENTUM,
    WEIGHT_DECAY,
    LEARN_RATE,
    PRINT_FREQ,
    WEIGHTS_PATH_DEFAULT,
    TRANSFORM_CIFAR10,
    TRANSFORM_SVHN,
    TRANSFORM_MNIST_TRAIN
)

from pathlib import Path
from models import (
    Model,
    get_model
)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.count = None
        self.sum = None
        self.avg = None
        self.val = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 after 8 and 14 epochs"""
    lr = lr * (0.1 ** (epoch // 8)) * (0.1 ** (epoch // 14))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def load_original_dataset(data_root: Path, batch_size, device=DEVICE, dataset_name="cifar10"):
    """
    Load an original dataset.
    :param data_root: The root path of all datasets.
    :param batch_size: The size of each batch.
    :param device: The device that the datasets should be on.
    :param transform: The transformation to apply on each dataset.
    :return: The training and testing datasets.
    """

    dl_kwargs = {'num_workers': 4, 'pin_memory': True} if device == "cuda" else {}
    dl_kwargs.update({"batch_size": batch_size, "shuffle": True})

    dset_kwargs = {"root": str(data_root), "download": True}
    dset_kwargs_maps = {
        "cifar10": {"transform": TRANSFORM_CIFAR10},
        "mnist": {"transform": TRANSFORM_MNIST_TRAIN},
        "svhn": {"transform": TRANSFORM_SVHN}
    }
    dset_kwargs.update(dset_kwargs_maps[dataset_name])

    # Python does eager evaluation, not lazy. So we didn't put this conditional in a dictionary.
    if dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(train=True, **dset_kwargs)
        test_dataset = datasets.CIFAR10(train=False, **dset_kwargs)
    elif dataset_name == "mnist":
        train_dataset = datasets.MNIST(train=True, **dset_kwargs)
        test_dataset = datasets.MNIST(train=False, **dset_kwargs)
    elif dataset_name == "svhn":
        train_dataset = ConcatDataset([datasets.SVHN(split="train", **dset_kwargs),
                                 datasets.SVHN(split="extra", **dset_kwargs)])
        test_dataset = datasets.SVHN(split="test", **dset_kwargs)
    else:
        raise Exception(f"No dataset named {dataset_name}.")

    train_loader = torch.utils.data.DataLoader(train_dataset, **dl_kwargs)

    dl_kwargs["shuffle"] = False
    test_loader = torch.utils.data.DataLoader(test_dataset, **dl_kwargs)

    return train_loader, test_loader


def train_epoch(train_loader: torch.utils.data.DataLoader,
                ss_batch_func,
                model,
                device,
                criterion,
                optimizer,
                epoch,
                print_freq):
    """
    :param train_loader: The training dataloader.
    :param ss_batch_func: Function to transform each batch from the training dataloader to suit the self-supervision task.
    :param model: The model to train.
    :param device: The device to run the datasets and model on.
    :param criterion: The criterion.
    :param optimizer: The optimiser.
    :param epoch: The epoch number.
    :param print_freq: The frequency at which the losses should be printed.
    :return: The losses
    """

    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    # top1 = AverageMeter()

    # As backbone is freezed, change it to evaluation mode
    model.eval()

    for i, (inp, _) in enumerate(train_loader):
        end = time.time()
        inp = inp.to(device)

        # Classification backbone frozen. Will always give 94.37% accuracy.
        # output, _ = model(inp)
        # loss_cls = criterion(output, target)

        # self-supervised
        inputs_ssh, labels_ssh = ss_batch_func(inp)
        inputs_ssh, labels_ssh = inputs_ssh.to(device), labels_ssh.to(device)
        _, outputs_ssh = model(inputs_ssh)
        loss_ssh = criterion(outputs_ssh, labels_ssh)

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
                  f"Time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
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

    for inputs, labels in test_dataloader:
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


def save_checkpoint(model: Model, weights_path: Path, is_best: bool, task_name: str, dataset_name: str):
    """
    Saves model weights to disk.
    :param model: The model.
    :param weights_path: The path to save the weights.
    :param is_best: Whether the current weights performed best
    :param task_name: The name of the task.
    :return: None.
    """
    directory = weights_path / model.model_name / dataset_name / task_name
    model.save_ss_fc(str(directory), "checkpoint.pt")

    if is_best:
        shutil.copyfile(str(directory / "checkpoint.pt"), str(directory / "best.pt"))


def train_ss_fc(
        model,
        device,
        train_loader,
        test_loader,
        ss_batch_func,
        ss_task_name,
        epochs=EPOCHS,
        learning_rate=LEARN_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        print_freq=PRINT_FREQ,
        show_animation=True,
        weights_path=WEIGHTS_PATH_DEFAULT,
        dataset_name="cifar10"):

    weights_path = Path(weights_path)

    # Set up figure for plotting losses per epoch
    figure = plt.figure()
    ax = figure.add_subplot(111)
    ax.set_title(f"Loss for each Epoch ({model.model_name})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    train_loss_x = []
    train_loss_y = []
    val_loss_x = []
    val_loss_y = []

    train_loss_plot = val_loss_plot = None
    if show_animation:
        train_loss_plot, = ax.plot(train_loss_x, train_loss_y, 'ro-', label="Train Loss")
        val_loss_plot, = ax.plot(val_loss_x, val_loss_y, "bx-", label="Val Loss")
        ax.legend()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                learning_rate,
                                momentum=momentum,
                                nesterov=True,
                                weight_decay=weight_decay)

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

        # evaluate on test set for self-supervised prediction
        ss_acc, val_losses = test_model(
            test_loader,
            model,
            device,
            ss_batch_func
        )

        # Update loss graph
        train_loss_x.append(epoch)
        train_loss_y.append(train_losses.avg)

        val_loss_x.append(epoch)
        val_loss_y.append(val_losses.avg)
        if show_animation:
            train_loss_plot.set_xdata(train_loss_x)
            train_loss_plot.set_ydata(train_loss_y)

            val_loss_plot.set_xdata(val_loss_x)
            val_loss_plot.set_ydata(val_loss_y)

            ax.relim()
            ax.autoscale_view(True, True, True)
            figure.canvas.draw()
            figure.canvas.flush_events()

        # remember best prec@1 and save checkpoint
        is_best = ss_acc > best_ss_acc
        best_ss_acc = max(ss_acc, best_ss_acc)
        save_checkpoint(
            model,
            weights_path,
            is_best,
            ss_task_name,
            dataset_name
        )

    if not show_animation:
        ax.plot(train_loss_x, train_loss_y, 'ro-', label="Train Loss")
        ax.plot(val_loss_x, val_loss_y, "bx-", label="Val Loss")

    figure.savefig(weights_path / model.model_name / dataset_name / ss_task_name / "train.png", format="png")
    plt.close("all")

    print(f'Best {ss_task_name} accuracy (test set):', best_ss_acc)


def train_original_dataset(dataset_name: str,
                           data_root: Path,
                           model_name: str,
                           num_ss_out: int,
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

    model = get_model(model_name, task_name, num_ss_out, device, False, dataset_name, force_reload=True)

    train_loader, test_loader = load_original_dataset(data_root, batch_size, device, dataset_name)

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
        weights_path=weights_path,
        dataset_name=dataset_name
    )

    if show_train_animation:
        plt.ioff()
        plt.show()