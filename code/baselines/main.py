import sys
sys.path.append(".")
from utils import (
    VALID_MODELS,
    DATA_PATH_DEFAULT,
    RESULTS_PATH_DEFAULT,
    ORIGINAL_DATASET_ROOT_DEFAULT,
    BATCH_SIZE,
    EPOCHS,
    LEARN_RATE,
    PRINT_FREQ,
    DEVICE,
    WEIGHTS_PATH_DEFAULT
)

import nuclear_norm, rotation, jigsaw
import rotation_invariance as ri
import jigsaw_invariance as ji
import img_classification as classification

if __name__ == "__main__":
    print(f"Currently using device: {DEVICE}")
    dataset = "cifar10"
    dsets = ["train_data", "val_data"]
    int_to_perm = jigsaw.construct_permutation_mappings(2, 4)
    for model_name in VALID_MODELS:
        print(f"Current model: {model_name}")
        print("Performing image classification.")
        classification.main(dataset, model_name, "classification", DATA_PATH_DEFAULT, RESULTS_PATH_DEFAULT, dsets, 4,
                            classification.calculate_acc)
        print("Performing rotation invariance.")
        ri.main(dataset, model_name, "rotation_invariance", DATA_PATH_DEFAULT, RESULTS_PATH_DEFAULT, dsets, 4,
                classification.calculate_acc)
        print("Performing jigsaw invariance.")

        ji.main(dataset,
                model_name,
                "jigsaw_invariance",
                DATA_PATH_DEFAULT,
                RESULTS_PATH_DEFAULT,
                dsets,
                4,
                lambda x, y, z: ji.jigsaw_inv_pred(x, y, z, int_to_perm, 2),
                recalculate_results=False)
        print("Performing nuclear norm.")
        nuclear_norm.main("cifar10", model_name, DATA_PATH_DEFAULT, RESULTS_PATH_DEFAULT,
                          ["train_data", "val_data"])
        print("Performing rotation prediction.")
        rotation.main(model_name,
                      DATA_PATH_DEFAULT,
                      ORIGINAL_DATASET_ROOT_DEFAULT,
                      dsets,
                      False,
                      BATCH_SIZE,
                      EPOCHS,
                      LEARN_RATE,
                      PRINT_FREQ,
                      False,
                      RESULTS_PATH_DEFAULT,
                      weights_path=WEIGHTS_PATH_DEFAULT,
                      dataset_name=dataset)
        print("Performing jigsaw prediction.")
        jigsaw.main(model_name,
                    DATA_PATH_DEFAULT,
                    ORIGINAL_DATASET_ROOT_DEFAULT,
                    dsets,
                    False,
                    BATCH_SIZE,
                    EPOCHS,
                    LEARN_RATE,
                    PRINT_FREQ,
                    False,
                    results_path=RESULTS_PATH_DEFAULT,
                    device=DEVICE,
                    weights_path=WEIGHTS_PATH_DEFAULT,
                    show_train_animation=False,
                    dataset_name=dataset,
                    grid_length=2,
                    max_perms=4)
