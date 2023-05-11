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

valid_models = [
    "resnet20",  # rotation, jigsaw done
    "resnet32",  # rotation, jigsaw done
    "resnet44",  # rotation, jigsaw done
    "resnet56",  # rotation, jigsaw done
    "resnet110",  # rotation, jigsaw done
    "resnet1202",  # rotation, jigsaw done
    "repvgg",  # rotation, jigsaw done
    "mobilenetv2",  # rotation, jigsaw done
    "densenet121",  # rotation, jigsaw done
    "densenet161",  # rotation, jigsaw done
    "densenet169",  # rotation, jigsaw done
    "shufflenet",  # rotation, jigsaw done
    "inceptionv3",  # rotation, jigsaw done
    "linear",  # rotation, jigsaw done
    "alexnet",  # rotation, jigsaw done
    "lenet5",  # rotation, jigsaw done
    "obc"  # rotation, jigsaw done.
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
        prob, _ = model(imgs)  # TODO: edit the models for other baselines.
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
    # TODO: Possibly shift this to the model class.
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


def fit_lr(train_x, train_y, val_x, val_y, task_name, model_name, show_graphs=False, save_graphs_dir=None):
    lr_train = LinearRegression()
    lr_train.fit(train_x.reshape(-1, 1), train_y)

    lr_val = LinearRegression()  # Linear regression for the validation set only.
    lr_val.fit(val_x.reshape(-1, 1), val_y)

    lr_train_train_y_hat = lr_train.predict(train_x.reshape(-1, 1))
    lr_train_val_y_hat = lr_train.predict(val_x.reshape(-1, 1))

    lr_val_val_y_hat = lr_val.predict(val_x.reshape(-1, 1))

    lr_train_rmse_loss_train = mean_squared_error(y_true=train_y, y_pred=lr_train_train_y_hat, squared=False)
    lr_train_rmse_loss_val = mean_squared_error(y_true=val_y, y_pred=lr_train_val_y_hat, squared=False)

    lr_val_rmse_loss_val = mean_squared_error(y_true=val_y, y_pred=lr_val_val_y_hat, squared=False)

    lr_train_r2_train = r2_score(train_y, lr_train_train_y_hat)
    lr_train_r2_val = r2_score(val_y, lr_train_val_y_hat)

    lr_val_r2_val = r2_score(val_y, lr_val_val_y_hat)

    print("Displaying Metrics")

    grid = [["Metric", "Training Set", "Validation Set (Val LR)", "Validation Set (Train LR)"],
            ["RMSE", f"{lr_train_rmse_loss_train:.4f}", f"{lr_val_rmse_loss_val:.4f}", f"{lr_train_rmse_loss_val:.4f}"],
            ["R^2", f"{lr_train_r2_train:.4f}", f"{lr_val_r2_val:.4f}", f"{lr_train_r2_val:.4f}"]]
    print(tabulate(grid, headers="firstrow", tablefmt="psql"))

    plt.figure()
    plt.title(f"Classification Acc. vs {task_name} Acc. ({model_name}) - Interior Domain")
    plt.xlabel(f"{task_name} Accuracy")
    plt.ylabel("Classification Accuracy")
    plt.scatter(train_x.reshape(-1, 1), train_y, marker=".")
    plt.plot(train_x.reshape(-1, 1), lr_train_train_y_hat, "r", label="Interior Domain Line")
    plt.legend(loc="best")
    if save_graphs_dir:
        plt.savefig(f"{save_graphs_dir}/{model_name}/{task_name}/original_cifar10_corrupted.png", format="png")

    plt.figure()
    plt.title(f"Classification Acc. vs {task_name} Acc. ({model_name}) - Exterior Domain")
    plt.xlabel(f"{task_name} Accuracy")
    plt.ylabel("Classification Accuracy")
    plt.scatter(val_x.reshape(-1, 1), val_y, marker=".")
    plt.plot(val_x.reshape(-1, 1), lr_train_val_y_hat, "r", label="Interior Domain Line")
    plt.plot(val_x.reshape(-1, 1), lr_val_val_y_hat, "g", label="Exterior Domain Line")
    plt.legend(loc="best")
    if save_graphs_dir:
        plt.savefig(f"{save_graphs_dir}/{model_name}/{task_name}/other_cifar10_corrupted.png", format="png")

    if show_graphs:
        plt.show()

    plt.close("all")
    return lr_train_rmse_loss_train, lr_val_rmse_loss_val, lr_train_rmse_loss_val, lr_train_r2_train, lr_val_r2_val, lr_train_r2_val
