import numpy as np
import torch.utils.data
import os
import shutil
import torch.nn.functional
import torchvision.transforms
from itertools import permutations


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

    return {k:{"perm": v, "inverse": inverse_permutation(v)} for k,v in enumerate(perms)}


def save_checkpoint(state, is_best, model_name, task):
    """Saves checkpoint to disk"""
    directory = f"../model_weights/{model_name}/{task}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = f"{directory}/checkpoint.pt"
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, f'../model_weights/{model_name}-{task}-fc.pt')


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 after 8 and 14 epochs"""
    lr = args.lr * (0.1 ** (epoch // 8)) * (0.1 ** (epoch // 14))
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