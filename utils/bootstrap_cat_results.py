from pathlib import Path
import json

import pandas as pd
from sklearn.metrics import roc_curve, auc
from multiprocessing import Pool, Manager
import numpy as np
from tqdm import tqdm

from utils.Utils import mean_confidence_interval
from sklearn.metrics import precision_score


def worker(
    data, i, tot, paths, auc_list_test, auc_list_train, prec_list_test, prec_list_train
):
    print(f"bootstrap results progress {i}/{tot}...")
    bootstrap = np.random.choice(paths, size=len(paths), replace=True)
    # loo_pairs = []
    # for path in bootstrap:
    #     all_files_minus_one = paths.copy()
    #     all_files_minus_one = np.delete(all_files_minus_one, paths.tolist().index(path))
    #     loo_pairs.append([path, all_files_minus_one])

    all_test_proba = []
    all_test_y = []
    all_train_proba = []
    all_train_y = []
    for filepath in bootstrap:
        # with open(filepath, 'r') as fp:
        # loo_result = json.load(fp)
        loo_result = data[filepath.stem]
        y_pred_proba_test = loo_result["y_pred_proba_test"]
        y_pred_proba_test = np.array(y_pred_proba_test)[:, 1]
        y_test = loo_result["y_test"]
        all_test_proba.extend(y_pred_proba_test)
        all_test_y.extend(y_test)

        y_pred_proba_train = loo_result["y_pred_proba_train"]
        y_pred_proba_train = np.array(y_pred_proba_train)[:, 1]
        y_train = loo_result["y_train"]
        all_train_proba.extend(y_pred_proba_train)
        all_train_y.extend(y_train)

    fpr, tpr, thresholds = roc_curve(all_test_y, all_test_proba)
    roc_auc = auc(fpr, tpr)
    auc_list_test.append(roc_auc)

    fpr, tpr, thresholds = roc_curve(all_train_y, all_train_proba)
    roc_auc = auc(fpr, tpr)
    auc_list_train.append(roc_auc)

    prec_list_test.append(
        precision_score(all_test_y, (np.array(all_test_proba) > 0.5).astype(int))
    )
    prec_list_train.append(
        precision_score(all_train_y, (np.array(all_train_proba) > 0.5).astype(int))
    )


def main(path=None, n_bootstrap=100, n_job=8):
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
            n_peaks = int(filepath.parent.parent.parent.parent.stem.split("__")[1])
            window_size = int(loo_result["training_shape"][1] / 60)
            clf = filepath.parent.parent.parent.stem[23:].split("_")[-1]
            if clf == "rbf":
                clf = "SVM(rbf)"
            if clf == "linear":
                clf = "SVM(linear)"
            pre_proc = "->".join(
                filepath.parent.parent.parent.stem[23:].split("_")[:-1]
            )

            data[filepath.stem] = {
                "y_pred_proba_test": loo_result["y_pred_proba_test"],
                "y_test": loo_result["y_test"],
                "y_pred_proba_train": loo_result["y_pred_proba_train"],
                "y_train": loo_result["y_train"],
                "training_size": training_size,
                "testing_size": testing_size,
                "n_peaks": n_peaks,
                "window_size": window_size,
                "clf": clf,
                "pre_proc": pre_proc,
            }

    print("start bootstrap...")
    pool = Pool(processes=n_job)
    with Manager() as manager:
        auc_list_test = manager.list()
        auc_list_train = manager.list()
        prec_list_test = manager.list()
        prec_list_train = manager.list()
        for i in range(n_bootstrap):
            pool.apply_async(
                worker,
                (
                    data,
                    i,
                    n_bootstrap,
                    paths,
                    auc_list_test,
                    auc_list_train,
                    prec_list_test,
                    prec_list_train,
                ),
            )
        pool.close()
        pool.join()
        pool.terminate()

        auc_list_test = list(auc_list_test)
        auc_list_train = list(auc_list_train)
        prec_list_test = list(prec_list_test)
        prec_list_train = list(prec_list_train)

    mean_auc_test = np.mean(auc_list_test)
    lo_test_auc, hi_test_auc = mean_confidence_interval(auc_list_test)
    print(
        f"Mean Test AUC = {mean_auc_test:.2f}, 95% CI [{lo_test_auc:.2f}, {hi_test_auc:.2f}] )"
    )

    mean_auc_train = np.mean(auc_list_train)
    lo_train_auc, hi_train_auc = mean_confidence_interval(auc_list_train)
    print(
        f"Mean Train AUC = {mean_auc_train:.2f}, 95% CI [{lo_train_auc:.2f}, {hi_train_auc:.2f}] )"
    )

    mean_prec_test = np.mean(prec_list_test)
    lo_test_prec, hi_test_prec = mean_confidence_interval(prec_list_test)
    print(
        f"Mean Test prec = {mean_prec_test:.2f}, 95% CI [{lo_test_prec:.2f}, {hi_test_prec:.2f}] )"
    )

    mean_prec_train = np.mean(prec_list_train)
    lo_train_prec, hi_train_prec = mean_confidence_interval(prec_list_train)
    print(
        f"Mean Train prec = {mean_prec_train:.2f}, 95% CI [{lo_train_prec:.2f}, {hi_train_prec:.2f}] )"
    )

    return [
        f"{mean_auc_test:.2f} ({lo_test_auc:.2f}-{hi_test_auc:.2f})",
        f"{mean_auc_train:.2f} ({lo_train_auc:.2f}-{hi_train_auc:.2f})",
        f"{mean_prec_test:.2f} ({lo_test_prec:.2f}-{hi_test_prec:.2f})",
        f"{mean_prec_train:.2f} ({lo_train_prec:.2f}-{hi_train_prec:.2f})",
        training_size,
        testing_size,
        n_peaks,
        window_size,
        clf,
        pre_proc,
        mean_auc_test,
        paths,
    ]


if __name__ == "__main__":
    results = []
    folders = [
        x
        for x in Path("E:/Cats/article/ml_build_permutations_qnf_final/").glob("*/*/*")
        if x.is_dir()
    ]
    for i, item in enumerate(folders):
        print(f"{i}/{len(folders)}...")
        res = main(Path(f"{item}/fold_data"))
        if res is not None:
            results.append(res)

    df = pd.DataFrame(
        results,
        columns=[
            "AUC testing (95% CI)",
            "AUC training (95% CI)",
            "Class1 Precision testing (95% CI)",
            "Class1 Precision training (95% CI)",
            "N training samples",
            "N testing samples",
            "N peaks",
            "Sample length (minutes)",
            "Classifier",
            "Pre-processing",
            "mean_auc_test",
            "path",
        ],
    )
    df_ = df.sort_values("mean_auc_test", ascending=False)
    df_ = df_.drop("mean_auc_test", axis=1)
    df_ = df_.drop("path", axis=1)
    df_ = df_.head(20)
    print(df_.to_latex(index=False))
    df.to_csv("cat_result_table.csv", index=False)
