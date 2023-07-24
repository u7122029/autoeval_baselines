import json
import os

from utils import (
    get_dirs,
    VALID_MODELS,
    TEMP_PATH_DEFAULT,
    DATA_PATH_DEFAULT,
    BATCH_SIZE,
    EPOCHS,
    LEARN_RATE,
    PRINT_FREQ,
    TRAIN_DATA,
    DEFAULT_DATASET_COND
)

import baselines.rotation as rotation, baselines.jigsaw as jigsaw, baselines.img_classification as classification

if __name__ == "__main__":
    d = json.loads(open("test_config.json", "r").read())
    for model_name, params in d.items():
        print(f"Current model: {model_name}")
        if model_name not in VALID_MODELS:
            print(f"Model name {model_name} not in valid_models. Skipping.")
            continue

        img_class_done = all([
            os.path.exists(f"{TEMP_PATH_DEFAULT}/{model_name}/classification/train_data.npy"),
            os.path.exists(f"{TEMP_PATH_DEFAULT}/{model_name}/classification/val_sets.npy")
        ])

        want_ss = any([params["rotation"], params["jigsaw"]])
        val_data = get_dirs(DATA_PATH_DEFAULT, DEFAULT_DATASET_COND)
        if (not img_class_done or params["img_classification"]) and want_ss:
            classification.main(model_name,
                                DATA_PATH_DEFAULT,
                                TEMP_PATH_DEFAULT,
                                True,
                                TRAIN_DATA,
                                val_data)

        if params["rotation"]:
            rotation.main(model_name,
                          DATA_PATH_DEFAULT,
                          params["show-graphs"],
                          params["train-ss-layer"],
                          BATCH_SIZE,
                          EPOCHS,
                          LEARN_RATE,
                          PRINT_FREQ,
                          params["eval-doms"],
                          False,
                          TRAIN_DATA,
                          val_data,
                          show_train_animation=False
                          )

        if params["jigsaw"]:
            pass #TODO: FIX THIS!
