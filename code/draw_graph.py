from pathlib import Path
from utils import (
    RESULTS_PATH_DEFAULT,
    VALID_MODELS,
    DATA_PATH_DEFAULT,
    ensure_cwd,
    VALID_TASK_NAMES,
    VALID_DATASETS
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from tabulate import tabulate

import numpy as np
import argparse

parser = argparse.ArgumentParser(description="AutoEval Baselines - Draw Graph")
parser.add_argument(
    "--model",
    required=True,
    type=str,
    help="The classifier to run.",
    choices=VALID_MODELS
)
parser.add_argument(
    "--data-root",
    required=False,
    default=DATA_PATH_DEFAULT,
    type=str,
    help="path containing all datasets (training and validation)"
)
parser.add_argument(
    "--dataset",
    required=False,
    default="cifar10",
    type=str,
    choices=VALID_DATASETS
)
parser.add_argument(
    "--dsets-train",
    required=True,
    nargs="*",
    help="List of relative training dataset roots, space separated."
)
parser.add_argument(
    "--dsets-val",
    required=True,
    nargs="*",
    help="List of relative validation dataset roots, space separated."
)
parser.add_argument(
    "--x-task",
    required=False,
    type=str,
    help="The name of the task on the x axis.",
    choices=VALID_TASK_NAMES,
    default="rotation"
)
parser.add_argument(
    "--y-task",
    required=False,
    type=str,
    help="The name of the task on the y axis.",
    choices=VALID_TASK_NAMES,
    default="classification"
)
parser.add_argument(
    "--results-root",
    required=False,
    default=RESULTS_PATH_DEFAULT,
    type=str,
    help="The root path to store the results."
)
parser.add_argument(
    "--output-path",
    required=False,
    type=str,
    default="figures/output.svg",
    help="The relative path to the diagram file."
)
parser.add_argument(
    "--y-transform",
    required=False,
    default="none",
    choices=["none", "exp", "log"],
    help="Whether the model should be an exponential fit."
)

EXPONENTIAL=0
LOGARITHMIC=1
y_transform_map = {
    "none": None,
    "exp": EXPONENTIAL,
    "log": LOGARITHMIC
}

def fit_lr(train_x,
           train_y,
           val_x,
           val_y,
           model_name: str,
           x_task: str,
           y_task: str,
           show_graphs: bool = False,
           results_root: Path = RESULTS_PATH_DEFAULT,
           output: Path = None,
           dataset_name="cifar10",
           y_transform: int = None):
    print(len(train_x), len(val_x))
    print(len(train_y), len(val_y))
    if y_transform == LOGARITHMIC:
        train_y = np.log(train_y)
        val_y = np.log(val_y)
        y_task = f"ln({y_task})"
    elif y_transform == EXPONENTIAL:
        train_y = np.exp(train_y)
        val_y = np.exp(val_y)
        y_task = f"exp({y_task})"
    lr_train = LinearRegression()
    lr_train.fit(train_x.reshape(-1, 1), train_y)

    lr_val = LinearRegression()  # Linear regression for the validation set only.
    lr_val.fit(val_x.reshape(-1, 1), val_y)

    lr_train_train_y_hat = lr_train.predict(train_x.reshape(-1, 1))
    lr_train_val_y_hat = lr_train.predict(val_x.reshape(-1, 1))

    lr_val_val_y_hat = lr_val.predict(val_x.reshape(-1, 1))

    # Scoring.
    lr_train_rmse_loss_train = mean_squared_error(y_true=train_y, y_pred=lr_train_train_y_hat, squared=False)
    lr_train_rmse_loss_val = mean_squared_error(y_true=val_y, y_pred=lr_train_val_y_hat, squared=False)
    lr_val_rmse_loss_val = mean_squared_error(y_true=val_y, y_pred=lr_val_val_y_hat, squared=False)

    lr_train_r2_train = r2_score(train_y, lr_train_train_y_hat)
    lr_train_r2_val = r2_score(val_y, lr_train_val_y_hat)
    lr_val_r2_val = r2_score(val_y, lr_val_val_y_hat)

    lr_train_sp_train = spearmanr(train_y, lr_train_train_y_hat).statistic
    lr_train_sp_val = spearmanr(val_y, lr_train_val_y_hat).statistic
    lr_val_sp_val = spearmanr(val_y, lr_val_val_y_hat).statistic

    print(f"Displaying Metrics - {dataset_name}")

    grid = [["Metric", "Training Set", "Validation Set (Val LR)", "Validation Set (Train LR)"],
            ["R^2", f"{lr_train_r2_train:.4f}", f"{lr_val_r2_val:.4f}", f"{lr_train_r2_val:.4f}"],
            ["rho", f"{lr_train_sp_train:.4f}", f"{lr_val_sp_val:.4f}", f"{lr_train_sp_val:.4f}"],
            ["MSE", f"{lr_train_rmse_loss_train:.4f}", f"{lr_val_rmse_loss_val:.4f}", f"{lr_train_rmse_loss_val:.4f}"]
            ]
    print(tabulate(grid, headers="firstrow", tablefmt="psql"))

    all_x = np.concatenate([train_x, val_x])
    all_pred_y_train = lr_train.predict(all_x.reshape(-1, 1))
    all_pred_y_val = lr_val.predict(all_x.reshape(-1, 1))

    title = f"{y_task} vs. {x_task} ({model_name}) - {dataset_name}"
    plt.figure()
    plt.title(title)
    plt.grid()
    plt.xlabel(f"{x_task}")
    plt.ylabel(y_task)
    plt.scatter(val_x.reshape(-1, 1), val_y, marker="x", alpha=0.5, linewidths=0.5, color="#f08080",
                label=r"$\bf{ExD}$" + " Datasets")
    plt.scatter(train_x.reshape(-1, 1), train_y, marker="+", alpha=0.5, linewidths=0.75, color="#A7C7E7",
                label=r"$\bf{InD}$" + " Datasets")
    plt.plot(all_x.reshape(-1, 1), all_pred_y_train, "b", label=r"$\bf{In}$" + "terior " + r"$\bf{D}$" + "omain Fit")
    plt.plot(all_x.reshape(-1, 1), all_pred_y_val, "r", label=r"$\bf{Ex}$" + "terior " + r"$\bf{D}$" + "omain Fit")

    plt.legend(loc="best")
    if output:
        str_transform = y_transform if y_transform is not None else ""
        p = results_root / "figures" / f"{model_name}_{x_task}{str_transform}.png"
        p.parents[0].mkdir(parents=True, exist_ok=True)
        plt.savefig(str(p), format="png")

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


def main(train_results_paths: list[str],
         val_results_paths: list[str],
         x_task: str,
         y_task: str,
         model_name: str,
         output_path: str,
         results_root: str = RESULTS_PATH_DEFAULT,
         show_graphs: bool = False,
         dataset_name: str = "cifar10",
         y_transform: str = None
         ):
    output_path = Path(output_path)
    results_root = Path(results_root)

    print(f"===> Linear Regression model for {y_task} vs {x_task} with model: {model_name}")
    train_x = np.concatenate(
        [np.array(np.load(str(results_root / "raw_findings" / dataset_name / train_results_path / f"{model_name}.npz"))
                  [x_task][:, 0], dtype=np.float64)# * 100.0
         for train_results_path in train_results_paths])

    train_y = np.concatenate(
        [np.array(np.load(str(results_root / "raw_findings" / dataset_name / train_results_path / f"{model_name}.npz"))
                  [y_task][:, 0], dtype=np.float64)# * 100.0
         for train_results_path in train_results_paths])

    val_x = np.concatenate(
        [np.array(np.load(str(results_root / "raw_findings" / dataset_name / val_results_path / f"{model_name}.npz"))
                  [x_task][:, 0], dtype=np.float64)# * 100.0
         for val_results_path in val_results_paths])

    val_y = np.concatenate(
        [np.array(np.load(str(results_root / "raw_findings" / dataset_name / val_results_path / f"{model_name}.npz"))
                  [y_task][:, 0], dtype=np.float64)# * 100.0
         for val_results_path in val_results_paths])


    fit_lr(train_x,
           train_y,
           val_x,
           val_y,
           model_name,
           x_task,
           y_task,
           show_graphs=show_graphs,
           results_root=results_root,
           output=output_path,
           dataset_name=dataset_name,
           y_transform=y_transform_map[y_transform])


if __name__ == "__main__":
    ensure_cwd()
    args = parser.parse_args()
    main(args.dsets_train,
         args.dsets_val,
         args.x_task,
         args.y_task,
         args.model,
         args.output_path,
         args.results_root,
         dataset_name=args.dataset,
         y_transform=args.y_transform)
