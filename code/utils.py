import os
import shutil
from itertools import permutations

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional
import torch.utils.data
import torchvision.transforms
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from tabulate import tabulate

TRANSFORM = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),
    ]
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


class CIFAR10NP(torch.utils.data.Dataset):
    # Dataset class for CIFAR10 dataset stored as numpy arrays
    def __init__(self, data_path, label_path, transform=None):
        # data_path and label_path are assumed to be ndarray objects
        self.imgs = np.load(data_path)
        # cast to int64 for model prediction
        self.labels = np.load(label_path).astype(dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def predict_multiple(model, imgs):
    # assume multiple image inputs with shape (N, 3, 32, 32) where N is the batch size
    assert isinstance(imgs, torch.Tensor) and imgs.shape[1:] == (3, 32, 32)
    # NOTE: make sure model is in validation mode
    model.eval()
    with torch.no_grad():
        prob = model(imgs)
        pred = prob.argmax(dim=1, keepdim=True)
    return pred, torch.nn.functional.softmax(prob, dim=1).cpu().numpy()


def store_ans(answers, file_name="answer.txt"):
    # This function ensures that the format of submission
    with open(file_name, "w") as f:
        for answer in answers:
            # Ensure that 6 decimals are used
            f.write("{:.6f}\n".format(answer))


def inverse_permutation(perm):
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.size(0), device=perm.device)
    return inv


def construct_permutation_mappings(grid_length):
    """
    Returns a mapping from the integers to grid_length**2 permutations in tuple form to the integers
    :param grid_length: The length of one side of the square grid
    :return: The integers to permutations mapping and its inverse.
    """
    perms = torch.tensor(list(permutations(range(grid_length ** 2))))

    return {k: {"perm": v, "inverse": inverse_permutation(v)} for k, v in enumerate(perms)}


def save_checkpoint(state, is_best, model_name, task):
    """Saves checkpoint to disk"""
    directory = f"../model_weights/{model_name}/{task}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = f"{directory}/checkpoint.pt"
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, f'../model_weights/{model_name}-{task}-fc.pt')


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 after 8 and 14 epochs"""
    lr = lr * (0.1 ** (epoch // 8)) * (0.1 ** (epoch // 14))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def fit_lr(train_x, train_y, val_x, val_y, show_graphs, task_name, model_name):
    lr = LinearRegression()
    lr.fit(train_x.reshape(-1, 1), train_y)

    train_y_hat = lr.predict(train_x.reshape(-1, 1))
    val_y_hat = lr.predict(val_x.reshape(-1, 1))

    rmse_loss_train = mean_squared_error(y_true=train_y, y_pred=train_y_hat, squared=False)
    rmse_loss_val = mean_squared_error(y_true=val_y, y_pred=val_y_hat, squared=False)

    r2_train = r2_score(train_y, train_y_hat)
    r2_val = r2_score(val_y, val_y_hat)

    print("Displaying Metrics")

    grid = [["Metric", "Training Set", "Validation Set"],
            ["RSME", f"{rmse_loss_train:.4f}", f"{rmse_loss_val:.4f}"],
            ["R^2", f"{r2_train:.4f}", f"{r2_val:.4f}"]]
    print(tabulate(grid, headers="firstrow", tablefmt="psql"))

    if show_graphs:
        plt.figure()
        plt.title(f"Classification Accuracy vs {task_name} Accuracy ({model_name}) - Training Dataset")
        plt.xlabel(f"{task_name} Accuracy")
        plt.ylabel("Classification Accuracy")
        plt.scatter(train_x.reshape(-1, 1), train_y, marker=".")
        plt.plot(train_x.reshape(-1, 1), train_y_hat, "r")

        plt.figure()
        plt.title(f"Classification Accuracy vs {task_name} Accuracy ({model_name}) - Validation Dataset")
        plt.xlabel(f"{task_name} Accuracy")
        plt.ylabel("Classification Accuracy")
        plt.scatter(val_x.reshape(-1, 1), val_y, marker=".")
        plt.plot(val_x.reshape(-1, 1), val_y_hat, "r")

        plt.show()
