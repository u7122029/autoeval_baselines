from pathlib import Path
from itertools import permutations

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional
import torch.utils.data
import torchvision.transforms as T
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from tabulate import tabulate
from models import Model
from tqdm import tqdm
from models import get_model

# PATHS
TEMP_PATH_DEFAULT = "../temp"
RESULTS_PATH_DEFAULT = "../results"
WEIGHTS_PATH_DEFAULT = "../model_weights"
DATA_PATH_DEFAULT = "data"
TRAIN_DATA = "train_data"
VAL_DATA = "val_data"
DEFAULT_DATASET_COND = lambda root, dirs, files: "data.npy" in files and "labels.npy" in files

# SS LAYER TRAINING
BATCH_SIZE = 64
JIGSAW_GRID_LENGTH = 2
PRINT_FREQ = 100
LEARN_RATE = 1e-2
EPOCHS = 25
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

TRANSFORM_CIFAR10 = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),
    ]
)

TRANSFORM_MNIST = T.Compose(
    [
        T.ToTensor()
    ]
)

TRANSFORM_SVHN = T.Compose(
    [
        T.ToTensor(),
        T.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
    ]
)

DSET_TRANSFORMS_EVAL = {
    "cifar10": TRANSFORM_CIFAR10,
    "mnist": TRANSFORM_MNIST,
    "svhn": TRANSFORM_SVHN
}


class ToRGB:
    def __init__(self):
        pass

    def __call__(self, sample):
        _input = sample
        return _input.repeat(3, 1, 1)


TRANSFORM_MNIST_TRAIN = T.Compose([T.Resize((32, 32)),
                                   T.ToTensor(),
                                   ToRGB()])

DSET_TRANSFORMS_TRAIN = {
    "cifar10": TRANSFORM_CIFAR10,
    "mnist": TRANSFORM_MNIST_TRAIN,
    "svhn": TRANSFORM_SVHN
}

VALID_MODELS = [
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
    "inception_v3",  # rotation, jigsaw done
    "linear",  # rotation, jigsaw done
    "alexnet",  # rotation, jigsaw done
    "lenet5",  # rotation, jigsaw done
]

VALID_DATASETS = [
    "cifar10",
    "mnist",
    "svhn"
]

VALID_TASK_NAMES = [
    "rotation",
    "classification"
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CIFAR10NP(torch.utils.data.Dataset):  # TODO: rename to DsetNP
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


class DatasetEvaluator:
    def __init__(self, dir_path: Path, transform):
        self.dir_path = dir_path
        self.transform = transform

    def evaluate(self, model: Model, predictor_func, device=DEVICE):
        """
        Evaluate the dataset over a given model and predictor function.
        :param model: The model.
        :param predictor_func: The predictor function.
        :param device: The device.
        :return: The accuracy of the model over the dataset.
        """
        data_path = str(self.dir_path / "data.npy")
        label_path = str(self.dir_path / "labels.npy")

        dataloader = torch.utils.data.DataLoader(
            dataset=CIFAR10NP(
                data_path=data_path,
                label_path=label_path,
                transform=self.transform,
            ),
            batch_size=500,
            shuffle=False,
        )

        acc = predictor_func(dataloader, model, device)
        return acc


def predict_multiple(model, imgs):
    # assume multiple image inputs with shape (N, 3, 32, 32) where N is the batch size
    assert isinstance(imgs, torch.Tensor) and imgs.shape[1:] == (3, 32, 32)
    # NOTE: make sure model is in validation mode
    model.eval()
    with torch.no_grad():
        prob, _ = model(imgs)
        pred = prob.argmax(dim=1, keepdim=True)
    return pred, torch.nn.functional.softmax(prob, dim=1).cpu().numpy()


def inverse_permutation(perm: torch.Tensor):
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.size(0), device=perm.device)
    return inv


def construct_permutation_mappings(grid_length, num_out_perms=None):
    """
    Returns a mapping from the integers to grid_length**2 permutations in tuple form to the integers
    :param grid_length: The length of one side of the square grid.
    :param num_out_perms: The number of output permutations.
    :return: The integers to permutations mapping and its inverse.
    """
    if num_out_perms is None:
        num_out_perms = grid_length ** 2

    perms = permutations(range(grid_length ** 2))
    out = {}
    for i in range(num_out_perms):
        perm = torch.Tensor(next(perms))
        out[i] = {"perm": perm, "inverse": inverse_permutation(perm)}

    return out


def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@model for the specified values of model
    :param output:
    :param target:
    :param topk:
    :return:
    """
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


def generate_results(dataset_name,
                     model_name,
                     task_name,
                     data_root,
                     results_path,
                     dset_paths,
                     model_ss_out_size,
                     predictor_func,
                     recalculate_results=False,
                     device=DEVICE,
                     load_best_fc=True):
    """
    Evaluates a given model over a specified task, storing the results in the given results path.
    :param model_name: The name of the model to load.
    :param task_name: The name of the task.
    :param data_root: The root path of all datasets.
    :param results_path: The root path of all results.
    :param dset_paths: The relative paths of each dataset.
    :param model_ss_out_size: The size of the self-supervised output layer.
    :param predictor_func: The predictor function.
    :param recalculate_results: True if the results should be recalculated, and False otherwise.
    :param device: The device the model and datasets should be run on.
    :param load_best_fc: True if the model should have its optimal weights loaded for a self-supervised task, and false
    otherwise (eg: for image classification).
    :return: None.
    """
    data_root = Path(data_root)
    results_path = Path(results_path)
    dset_paths = [Path(i) for i in dset_paths]

    # load the model
    model = get_model(model_name, task_name, model_ss_out_size, device, load_best_fc=load_best_fc,
                      dataset_name=dataset_name)
    model.eval()

    for dset_collection_root in dset_paths:
        results_root = results_path / "raw_findings" / dataset_name / dset_collection_root
        dataset_recurse(data_root / dataset_name / dset_collection_root,
                        results_root,
                        task_name,
                        model,
                        predictor_func,
                        device,
                        recalculate_results,
                        dataset_name=dataset_name)


def r2_adjusted(score, n, f):
    """
    Computes the R^2 adjusted score.
    :param score: The original R^2 score
    :param n: The number of datapoints.
    :param f: The number of independent variables.
    :return: The adjusted R^2 score.
    """
    return 1 - (1 - score) * (n - 1) / (n - f - 1)


def fit_lr(train_x,
           train_y,
           val_x,
           val_y,
           title: str,
           x_task: str,
           y_task: str,
           show_graphs: bool = False,
           results_root: Path = RESULTS_PATH_DEFAULT,
           output: Path = None,
           dataset_name="cifar10"):
    print(train_x.shape, train_y.shape)
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

    # Scoring.
    lr_train_r2_train = r2_score(train_y, lr_train_train_y_hat)
    lr_train_r2_val = r2_score(val_y, lr_train_val_y_hat)
    lr_val_r2_val = r2_score(val_y, lr_val_val_y_hat)

    lr_train_r2_train_adjusted = r2_adjusted(lr_train_r2_train, len(train_y), 1)
    lr_train_r2_val_adjusted = r2_adjusted(lr_train_r2_val, len(val_y), 1)
    lr_val_r2_val_adjusted = r2_adjusted(lr_val_r2_val, len(val_y), 1)

    print(f"Displaying Metrics - {dataset_name}")

    grid = [["Metric", "Training Set", "Validation Set (Val LR)", "Validation Set (Train LR)"],
            ["RMSE", f"{lr_train_rmse_loss_train:.4f}", f"{lr_val_rmse_loss_val:.4f}", f"{lr_train_rmse_loss_val:.4f}"],
            ["R^2", f"{lr_train_r2_train:.4f}", f"{lr_val_r2_val:.4f}", f"{lr_train_r2_val:.4f}"],
            ["R^2 Adjusted", f"{lr_train_r2_train_adjusted:.4f}", f"{lr_val_r2_val_adjusted:.4f}",
             f"{lr_train_r2_val_adjusted:.4f}"]]
    print(tabulate(grid, headers="firstrow", tablefmt="psql"))

    all_x = np.concatenate([train_x, val_x])
    all_pred_y_train = lr_train.predict(all_x.reshape(-1, 1))
    all_pred_y_val = lr_val.predict(all_x.reshape(-1, 1))

    plt.figure()
    plt.title(title)
    plt.grid()
    plt.xlabel(f"{x_task.capitalize()} Accuracy")
    plt.ylabel(f"{y_task.capitalize()} Accuracy")
    plt.scatter(val_x.reshape(-1, 1), val_y, marker="x", alpha=0.5, linewidths=0.5, color="#f08080",
                label=r"$\bf{ExD}$" + " Datasets")
    plt.scatter(train_x.reshape(-1, 1), train_y, marker="+", alpha=0.5, linewidths=0.75, color="#A7C7E7",
                label=r"$\bf{InD}$" + " Datasets")
    plt.plot(all_x.reshape(-1, 1), all_pred_y_train, "b", label=r"$\bf{In}$" + "terior " + r"$\bf{D}$" + "omain Fit")
    plt.plot(all_x.reshape(-1, 1), all_pred_y_val, "r", label=r"$\bf{Ex}$" + "terior " + r"$\bf{D}$" + "omain Fit")

    plt.legend(loc="best")
    if output:
        p = results_root / output
        p.parents[0].mkdir(parents=True, exist_ok=True)
        plt.savefig(str(results_root / output), format="svg")

    if show_graphs:
        plt.show()

    plt.close("all")
    return (lr_train_rmse_loss_train,
            lr_val_rmse_loss_val,
            lr_train_rmse_loss_val,
            lr_train_r2_train,
            lr_val_r2_val,
            lr_train_r2_val
            )


def dataset_recurse(data_root: Path, temp_root: Path, name: str, model: Model, predictor_func, device=DEVICE,
                    recalculate=False, dataset_name="cifar10"):
    outfile_path = temp_root / f"{model.model_name}.npz"
    #if not recalculate and outfile_path.exists() and name in np.load(str(outfile_path)):
    #    return

    is_leaf = (data_root / "data.npy").exists() and (data_root / "labels.npy").exists()
    leaf_result_exists = outfile_path.exists() and name in np.load(str(outfile_path))
    if is_leaf and ((not leaf_result_exists) or recalculate):
        # Leaf directory. Ignore anything else in here.
        evaluator = DatasetEvaluator(data_root, DSET_TRANSFORMS_EVAL[dataset_name])
        acc = evaluator.evaluate(model, predictor_func, device)

        temp_root.mkdir(parents=True, exist_ok=True)

        data = {}
        if outfile_path.exists():
            data = dict(np.load(str(outfile_path)))
        data[name] = np.array([(acc, str(data_root))])
        np.savez(str(outfile_path), **data)
        return

    if is_leaf:
        return

    # Otherwise we are in a subdirectory.
    # Visit all subdirs.
    out = []
    dirs = sorted(data_root.iterdir())  # Sorting ensures order.
    progressbar = tqdm(dirs, total=len(list(dirs)))
    for path in progressbar:
        progressbar.set_postfix({"data_dir": str(data_root), "temp_dir": str(temp_root)})
        if not path.is_dir():
            # Skip files.
            continue
        entity = path.parts[-1]
        dataset_recurse(data_root / entity, temp_root / entity, name, model, predictor_func, device, recalculate,
                        dataset_name=dataset_name)
        loaded = np.load(str(temp_root / entity / f"{model.model_name}.npz"))[name]
        out.append(loaded)
    out = np.concatenate(out)

    temp_root.mkdir(parents=True, exist_ok=True)
    data = {}
    if outfile_path.exists():
        data = dict(np.load(str(outfile_path)))
    data[name] = out
    np.savez(str(outfile_path), **data)


def ensure_cwd():
    current_dir = Path.cwd()
    checker = current_dir.parts
    if checker[-1] != "code" and checker[-2] != "autoeval_baselines":
        raise Exception(
            f"STOP. Make sure you are running this program in autoeval_baselines/code.\n"
            f"You are currently running this program in {current_dir}. Check the README.md for more information.")


if __name__ == "__main__":
    pass
