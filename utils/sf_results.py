from pathlib import Path
import numpy as np
from tqdm import tqdm
import json


def main(path):
    print("loading data...")
    paths = np.array(list(path.glob("*.json")))
    if len(paths) == 0:
        return
    data = {}
    for filepath in tqdm(paths):
        with open(filepath, "r") as fp:
            try:
                loo_result = json.load(fp)
            except Exception as e:
                return
        training_size = loo_result["training_shape"][0]
        testing_size = loo_result["testing_shape"][0]


if __name__ == "__main__":
    results = []
    folders = [x for x in Path("H:/thesis_final_feb16/thesis_final_feb16/main_experiment").glob('*/*/*/*') if x.is_dir()]
    for i, item in enumerate(folders):
        fold_data = item / "fold_data"
        print(f"{i}/{len(folders)}...")
        if not fold_data.exists():
            continue
        res = main(fold_data)

