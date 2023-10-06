from pathlib import Path
from utils import (
    RESULTS_PATH_DEFAULT, fit_lr, VALID_MODELS, DATA_PATH_DEFAULT, ensure_cwd, VALID_TASK_NAMES, VALID_DATASETS
)

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


def main(train_results_paths: list[str],
         val_results_paths: list[str],
         x_task: str,
         y_task: str,
         model_name: str,
         output_path: str,
         results_root: str = RESULTS_PATH_DEFAULT,
         show_graphs: bool = False,
         dataset_name: str = "cifar10"
         ):
    output_path = Path(output_path)
    results_root = Path(results_root)

    print(f"===> Linear Regression model for {y_task} vs {x_task} with model: {model_name}")
    train_x = np.concatenate(
        [np.array(np.load(str(results_root / "raw_findings" / dataset_name / train_results_path / f"{model_name}.npz"))[
                      x_task][:, 0], dtype=np.float64) * 100.0
         for train_results_path in train_results_paths])

    train_y = np.concatenate(
        [np.array(np.load(str(results_root / "raw_findings" / dataset_name / train_results_path / f"{model_name}.npz"))[
                      y_task][:, 0], dtype=np.float64) * 100.0
         for train_results_path in train_results_paths])

    val_x = np.concatenate(
        [np.array(
            np.load(str(results_root / "raw_findings" / dataset_name / val_results_path / f"{model_name}.npz"))[x_task][
            :, 0], dtype=np.float64) * 100.0
         for val_results_path in val_results_paths])

    val_y = np.concatenate(
        [np.array(
            np.load(str(results_root / "raw_findings" / dataset_name / val_results_path / f"{model_name}.npz"))[y_task][
            :, 0], dtype=np.float64) * 100.0
         for val_results_path in val_results_paths])

    title = f"{y_task.capitalize()} vs. {x_task.capitalize()} ({model_name}) - {dataset_name}"

    fit_lr(train_x,
           train_y,
           val_x,
           val_y,
           title,
           x_task,
           y_task,
           show_graphs=show_graphs,
           results_root=results_root,
           output=output_path,
           dataset_name=dataset_name)


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
         dataset_name=args.dataset)
