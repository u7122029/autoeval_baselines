from training_utils import get_model
from utils import VALID_MODELS
import pandas as pd

if __name__ == "__main__":
    d = {"model_name": [], "params": []}

    for model_name in VALID_MODELS:
        model = get_model(model_name, "rotation", 4, "cpu")
        d["model_name"].append(model_name)
        d["params"].append(len(list(model.model.parameters())))

    compiled = pd.DataFrame(d).sort_values("params")
    print(compiled)