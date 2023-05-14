from training_utils import get_model
from utils import valid_models
import pandas as pd

if __name__ == "__main__":
    d = {"model_name": [], "params": []}

    for model_name in valid_models:
        model = get_model(model_name, "rotation", 4, "cpu")
        d["model_name"].append(model_name)
        d["params"].append(len(list(model.model.parameters())))

    compiled = pd.DataFrame(d).sort_values("params")
    print(compiled)