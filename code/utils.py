from pathlib import Path
from itertools import permutations
from math import factorial
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional
import torch.utils.data
import torchvision.transforms as T
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from tabulate import tabulate
from models import Model, get_model
from tqdm import tqdm

# PATHS
ORIGINAL_DATASET_ROOT_DEFAULT = "C:/ml_datasets" # Path to original version of cifar10, svhn, mnist, etc
RESULTS_PATH_DEFAULT = "../results"
WEIGHTS_PATH_DEFAULT = "../model_weights"
DATA_PATH_DEFAULT = "data"
TRAIN_DATA = "train_data"
VAL_DATA = "val_data"
DEFAULT_MAX_JIGSAW_PERMS = 4
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
    "shufflenet",  # rotation, jigsaw done
    "inception_v3",  # rotation, jigsaw done
    "repvgg",  # rotation, jigsaw done
    "resnet110",  # rotation, jigsaw done
    "googlenet", # rotation, jigsaw done.
    "densenet161",  # rotation, jigsaw done
    "densenet169",  # rotation, jigsaw done
    "resnet1202",  # rotation, jigsaw done
    "mobilenetv2",  # rotation, jigsaw done
    "densenet121",  # rotation, jigsaw done
    "resnet32",  # rotation, jigsaw done
    "linear",  # rotation, jigsaw done
    "resnet44",  # rotation, jigsaw done
    "resnet56",  # rotation, jigsaw done
    "lenet5"  # rotation, jigsaw done
]

VALID_DATASETS = [
    "cifar10",
    "mnist",
    "svhn"
]

VALID_TASK_NAMES = [
    "rotation",
    "classification",
    "nuclear_norm",
    "rotation_invariance",
    "jigsaw_invariance",
    "jigsaw-grid-len-2_max-perm-4"
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

    def evaluate(self, model: Model, predictor_func, device=DEVICE, batch_size=BATCH_SIZE):
        """
        Evaluate the dataset over a given model and predictor function.
        :param model: The model.
        :param predictor_func: The predictor function.
        :param device: The device.
        :return: The accuracy of the model over the dataset.
        """
        data_path = str(self.dir_path / "data.npy")
        label_path = str(self.dir_path / "labels.npy")

        dset = CIFAR10NP(data_path=data_path, label_path=label_path, transform=self.transform)
        dataloader = torch.utils.data.DataLoader(
            dset,
            batch_size=batch_size if batch_size is not None else len(dset),
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
    return pred, torch.nn.functional.softmax(prob, dim=1)


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
                     load_best_fc=True,
                     batch_size=BATCH_SIZE):
    """
    Evaluates a given model over a specified task, storing the results in the given results path.
    :param model_name: The name of the model to load.
    :param task_name: The name of the task.
    :param data_root: The root path of all datasets.
    :param results_path: The root path of all results.
    :param dset_paths: The relative paths of each dataset.
    :param model_ss_out_size: The size of the self-supervised output layer.
    :param predictor_func: The predictor function. Takes in a dataloader, model and device. Outputs a metric such
                            as accuracy or effective invariance (EI)
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
    model = get_model(model_name, task_name, model_ss_out_size, device,
                      load_best_fc=load_best_fc,
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
                        dataset_name=dataset_name,
                        batch_size=batch_size)


def dataset_recurse(data_root: Path, temp_root: Path, name: str, model: Model, predictor_func, device=DEVICE,
                    recalculate=False, dataset_name="cifar10", batch_size=BATCH_SIZE):
    """

    :param data_root:
    :param temp_root:
    :param name:
    :param model:
    :param predictor_func:
    :param device:
    :param recalculate:
    :param dataset_name:
    :return:
    """
    outfile_path = temp_root / f"{model.model_name}.npz"
    #if not recalculate and outfile_path.exists() and name in np.load(str(outfile_path)):
    #    return

    is_leaf = (data_root / "data.npy").exists() and (data_root / "labels.npy").exists()
    leaf_result_exists = outfile_path.exists() and name in np.load(str(outfile_path))
    if is_leaf and ((not leaf_result_exists) or recalculate):
        # Leaf directory. Ignore anything else in here.
        evaluator = DatasetEvaluator(data_root, DSET_TRANSFORMS_EVAL[dataset_name])
        score = evaluator.evaluate(model, predictor_func, device, batch_size)

        temp_root.mkdir(parents=True, exist_ok=True)

        data = {}
        if outfile_path.exists():
            data = dict(np.load(str(outfile_path)))
        data[name] = np.array([(score, str(data_root))])
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
                        dataset_name=dataset_name, batch_size=batch_size)
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
    print(f"Currently using device: {DEVICE}")
    current_dir = Path.cwd()
    checker = current_dir.parts
    if checker[-1] != "code" and checker[-2] != "autoeval_baselines":
        raise Exception(
            f"STOP. Make sure you are running this program in autoeval_baselines/code.\n"
            f"You are currently running this program in {current_dir}. Check the README.md for more information.")


if __name__ == "__main__":
    pass
