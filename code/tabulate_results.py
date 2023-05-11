from tabulate import tabulate
import os
import argparse
import torch
from utils import fit_lr, DEVICE, TRANSFORM, valid_models
from training_utils import test_model, load_original_cifar_dataset
from eval_utils import eval_validation
from models.obc import OBC_SS
from baselines.rotation import rotate_batch, rotation_pred
import numpy as np
import pandas as pd

if __name__ == "__main__":
    temp_file_path = "../temp"

    train_dataloader, test_dataloader = load_original_cifar_dataset(DEVICE, 64, "data")
    val_sets = sorted(["cifar10-f-32", "cifar-10.1-c", "cifar-10.1"])
    df_dict = {
        "rotation": {
            "model": [],
            "R^2 (Interior Domain LR)": [],
            "RMSE (Interior Domain LR)": [],
            "R^2 (Exterior Domain LR)": [],
            "RMSE (Exterior Domain LR)": [],
            "R^2 (Exterior Domain w/ Interior Domain LR)": [],
            "RMSE (Exterior Domain w/ Interior Domain LR)": [],
        },
        "jigsaw": {
            "model": [],
            "R^2 (Interior Domain LR)": [],
            "RMSE (Interior Domain LR)": [],
            "R^2 (Exterior Domain LR)": [],
            "RMSE (Exterior Domain LR)": [],
            "R^2 (Exterior Domain w/ Interior Domain LR)": [],
            "RMSE (Exterior Domain w/ Interior Domain LR)": [],
        }
    }
    for model_name in valid_models:
        for task_name in ["rotation", "jigsaw"]:
            train_x = np.load(f"{temp_file_path}/{model_name}/{task_name}/train_data.npy") * 100
            train_y = np.load(f"{temp_file_path}/{model_name}/acc/train_data.npy") * 100
            val_x = np.load(f"{temp_file_path}/{model_name}/{task_name}/val_sets.npy") * 100
            val_y = np.load(f"{temp_file_path}/{model_name}/acc/val_sets.npy") * 100

            lr_internal_rmse_internal, \
                lr_external_rmse_external, \
                lr_internal_rmse_external, \
                lr_internal_r2_internal, \
                lr_external_r2_external, \
                lr_internal_r2_external = \
                fit_lr(train_x, train_y, val_x, val_y, task_name, model_name, save_graphs_dir=temp_file_path)

            df_dict[task_name]["model"].append(model_name)
            df_dict[task_name]["R^2 (Interior Domain LR)"].append(lr_internal_r2_internal)
            df_dict[task_name]["RMSE (Interior Domain LR)"].append(lr_internal_rmse_internal)
            df_dict[task_name]["R^2 (Exterior Domain LR)"].append(lr_external_r2_external)
            df_dict[task_name]["RMSE (Exterior Domain LR)"].append(lr_external_rmse_external)
            df_dict[task_name]["R^2 (Exterior Domain w/ Interior Domain LR)"].append(lr_internal_r2_external)
            df_dict[task_name]["RMSE (Exterior Domain w/ Interior Domain LR)"].append(lr_internal_rmse_external)

    rotation_df_full = pd.DataFrame(df_dict["rotation"]).round(4)
    rotation_df_interior = rotation_df_full[["model", "R^2 (Interior Domain LR)", "RMSE (Interior Domain LR)"]]\
        .sort_values("R^2 (Interior Domain LR)", ascending=False)
    rotation_df_exterior = rotation_df_full[["model", "R^2 (Exterior Domain LR)", "RMSE (Exterior Domain LR)"]]\
        .sort_values("R^2 (Exterior Domain LR)", ascending=False)
    rotation_df_exterior_w_interior = rotation_df_full[
        ["model", "R^2 (Exterior Domain w/ Interior Domain LR)", "RMSE (Exterior Domain w/ Interior Domain LR)"]]\
        .sort_values("R^2 (Exterior Domain w/ Interior Domain LR)", ascending=False)

    jigsaw_df_full = pd.DataFrame(df_dict["jigsaw"]).round(4)
    jigsaw_df_interior = jigsaw_df_full[["model", "R^2 (Interior Domain LR)", "RMSE (Interior Domain LR)"]] \
        .sort_values("R^2 (Interior Domain LR)", ascending=False)
    jigsaw_df_exterior = jigsaw_df_full[["model", "R^2 (Exterior Domain LR)", "RMSE (Exterior Domain LR)"]] \
        .sort_values("R^2 (Exterior Domain LR)", ascending=False)
    jigsaw_df_exterior_w_interior = jigsaw_df_full[
        ["model", "R^2 (Exterior Domain w/ Interior Domain LR)", "RMSE (Exterior Domain w/ Interior Domain LR)"]] \
        .sort_values("R^2 (Exterior Domain w/ Interior Domain LR)", ascending=False)

    # Store tabulated versions of dataframes.
    results_dir = f"{temp_file_path}/results"
    items = {
        "rotation_t_interior": tabulate(rotation_df_interior, showindex="never", headers="keys", tablefmt="github"),
        "rotation_t_exterior": tabulate(rotation_df_exterior, showindex="never", headers="keys", tablefmt="github"),
        "rotation_t_exterior_w_interior": tabulate(rotation_df_exterior_w_interior,
                                              showindex="never", headers="keys", tablefmt="github"),

        "jigsaw_t_interior": tabulate(jigsaw_df_interior, showindex="never", headers="keys", tablefmt="github"),
        "jigsaw_t_exterior": tabulate(jigsaw_df_exterior, showindex="never", headers="keys", tablefmt="github"),
        "jigsaw_t_exterior_w_interior": tabulate(jigsaw_df_exterior_w_interior,
                                                 showindex="never", headers="keys", tablefmt="github")
    }

    for task_name in ["rotation", "jigsaw"]:
        if not os.path.exists(f"{results_dir}/{task_name}"):
            os.makedirs(f"{results_dir}/{task_name}")

        for fname in ["interior", "exterior", "exterior_w_interior"]:
            f = open(f"{results_dir}/{task_name}/{fname}.txt", "w")
            f.write(str(items[f"{task_name}_t_{fname}"]))
            f.close()