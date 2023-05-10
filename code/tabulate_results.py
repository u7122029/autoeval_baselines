from tabulate import tabulate
import argparse
import torch
from utils import fit_lr, DEVICE, TRANSFORM
from training_utils import test_model, load_original_cifar_dataset
from eval_utils import eval_validation
from models.obc import OBC_SS
from baselines.rotation import rotate_batch, rotation_pred
import numpy as np

if __name__ == "__main__":
    temp_file_path = "../temp/obc"
    train_dataloader, test_dataloader = load_original_cifar_dataset(DEVICE,64,"data")

    model = OBC_SS(device=DEVICE).to(DEVICE)
    state_dict = torch.load("../model_weights/obc-rotation-fc.pt")
    model.load_state_dict(state_dict)

    ss_batch_func = lambda inp_batch: rotate_batch(inp_batch, "rand")
    test_model(test_dataloader,model,DEVICE,ss_batch_func)

    val_sets = sorted(["cifar10-f-32", "cifar-10.1-c", "cifar-10.1"])
    ss_predictor_func = lambda dataloader: rotation_pred(dataloader, model, DEVICE)
    eval_validation("data", "../temp", val_sets, TRANSFORM, 64, ss_predictor_func, save_results=False)

    train_x = np.load(f"{temp_file_path}/rotation/train_data.npy") * 100
    train_y = np.load(f"{temp_file_path}/acc/train_data.npy") * 100
    val_x = np.load(f"{temp_file_path}/rotation/val_sets.npy") * 100
    val_y = np.load(f"{temp_file_path}/acc/val_sets.npy") * 100
