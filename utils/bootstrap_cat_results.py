from pathlib import Path
import json
from sklearn.metrics import roc_curve, auc
from multiprocessing import Pool, Manager
import numpy as np
from tqdm import tqdm

from utils.Utils import mean_confidence_interval


def worker(data, i, tot, paths, auc_list_test, auc_list_train):
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


def main(path=None, n_bootstrap=1000, n_job=6):
    print("loading data...")
    paths = np.array(list(path.glob("*.json")))
    data = {}
    for filepath in tqdm(paths):
        with open(filepath, "r") as fp:
            loo_result = json.load(fp)
            data[filepath.stem] = {
                "y_pred_proba_test": loo_result["y_pred_proba_test"],
                "y_test": loo_result["y_test"],
                "y_pred_proba_train": loo_result["y_pred_proba_train"],
                "y_train": loo_result["y_train"],
            }

    print("start bootstrap...")
    pool = Pool(processes=n_job)
    with Manager() as manager:
        auc_list_test = manager.list()
        auc_list_train = manager.list()
        for i in range(n_bootstrap):
            pool.apply_async(
                worker,
                (data, i, n_bootstrap, paths, auc_list_test, auc_list_train),
            )
        pool.close()
        pool.join()
        pool.terminate()

        auc_list_test = list(auc_list_test)
        auc_list_train = list(auc_list_train)

    mean_auc_test = np.mean(auc_list_test)
    lo_test, hi_test = mean_confidence_interval(auc_list_test)
    print(
        f"Mean Test AUC = {mean_auc_test:.2f}, 95% CI [{lo_test:.2f}, {hi_test:.2f}] )"
    )

    mean_auc_train = np.mean(auc_list_train)
    lo_train, hi_train = mean_confidence_interval(auc_list_train)
    print(
        f"Mean Train AUC = {mean_auc_train:.2f}, 95% CI [{lo_train:.2f}, {hi_train:.2f}] )"
    )

    return mean_auc_test, lo_test, hi_test, mean_auc_train, lo_train, hi_train


if __name__ == "__main__":
    main(
        Path(
            "E:/Cats/article/ml_build_permutations_qnf_final/1000__003__0_00100__030/cats_LeaveOneOut_-1_-1_QN_rbf/0.0__1.0/fold_data"
        )
    )
