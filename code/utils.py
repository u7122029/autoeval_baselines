import numpy as np
import torch.utils.data
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

def construct_permutation_mappings(grid_length):
    """
    Returns a mapping from grid_length**2 permutations in tuple form to the integers in [0,grid_length^2), as well
    as its inverse.
    :param grid_length: The length of one side of the square grid
    :return: The permutations to integers mapping and its inverse.
    """
    return {k:v for v,k in enumerate(list(permutations(range(grid_length ** 2))))}, \
            {k:v for k,v in enumerate(list(permutations(range(grid_length ** 2))))}