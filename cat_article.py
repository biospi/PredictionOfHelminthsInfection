import pandas as pd
from sklearn.metrics import (
    auc,
    roc_curve,
)
from sklearn.svm import SVC
import numpy as np

if __name__ == "__main__":
    path = 'E:/Cats/build_permutations/800__001__0_00100__120/dataset/training_sets/samples/samples.csv'
    df = pd.read_csv(path, header=None)
    header = list(df.columns.values)
    header[-9] = "label"
    header[-8] = "id"
    header[-7] = "imputed_days"
    header[-6] = "date"
    header[-5] = "health"
    header[-4] = "target"
    header[-3] = "age"
    header[-2] = "name"
    header[-1] = "mobility_score"

    df.columns = header
    data_fold = []
    for cat in df["id"].unique():
        df_test = df[df["id"] == cat].iloc[:, :-8]
        df_train = df[df["id"] != cat].iloc[:, :-8]
        data_fold.append([df_test, df_train, df[df["id"] == cat]["name"].values[0]])

    for kernel in ["linear", "rbf"]:
        all_y = []
        all_y_pred_test = []
        all_y_pred_train = []
        for i, fold in enumerate(data_fold):
            X_train = fold[1].iloc[:, :-1]
            y_train = fold[1]['label']
            X_test = fold[0].iloc[:, :-1]
            y_test = fold[0]['label']
            clf = SVC(kernel=kernel, probability=True)
            print("fitting...")
            clf.fit(X_train, y_train)
            y_pred_test = clf.predict_proba(X_test)
            y_pred_train = clf.predict_proba(X_train)

            all_y.extend(y_test)
            all_y_pred_test.extend(y_pred_test)
            all_y_pred_train.extend(y_pred_train)
            print(f"FOLD {i}, cat {fold[2]}, kernel {kernel}:")
            print(f"y      --> {y_test.values}")
            print(f"y_pred_test --> {y_pred_test}")
            print(f"y_pred_train --> {y_pred_train}")
            print(f"training y balance -->\n {y_train.value_counts()}")
            print(f"testing y balance -->\n {y_test.value_counts()}")
            print("***************************************************")

        all_y = np.array(all_y)
        all_y_pred_test = np.array(all_y_pred_test)
        all_y_pred_test = all_y_pred_test[:, 1]
        all_y_pred_train = np.array(all_y_pred_train)
        all_y_pred_train = all_y_pred_train[:, 1]

        fpr, tpr, thresholds = roc_curve(all_y, all_y_pred_test)
        roc_auc = auc(fpr, tpr)
        print(all_y)
        print(all_y_pred_test)
        print(f"AUC TEST={roc_auc}")

        fpr, tpr, thresholds = roc_curve(all_y, all_y_pred_train)
        roc_auc = auc(fpr, tpr)
        print(all_y)
        print(all_y_pred_train)
        print(f"AUC TRAIN={roc_auc}")


