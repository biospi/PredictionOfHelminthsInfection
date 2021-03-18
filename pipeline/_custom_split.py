import random

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import LeavePOut, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score, balanced_accuracy_score, precision_score, f1_score
from sklearn.metrics import auc
import warnings
from sklearn import datasets

np.random.seed(0)


class StratifiedLeaveTwoOut:
    def __init__(self, animal_ids, n_repeats=1):
        self.n_repeats = n_repeats
        self.animal_ids = animal_ids

    def split(self, X, y, verbose=True, group=None):
        y = y.reshape(y.size, 1)
        animal_ids = self.animal_ids.reshape(self.animal_ids.size, 1)
        df = pd.DataFrame(np.hstack((np.hstack((X, y)), animal_ids)))
        header = ["feature_" + str(x) for x in df.columns.tolist()]
        header[-1] = "id"
        header[-2] = "target"
        df.columns = header
        training_idx = []
        testing_idx = []
        if verbose:
            print(df)
        for i in range(self.n_repeats):
            if df.shape[0] == 0:
                warnings.warn("cannot build more folds MAX REPEAT=%d" % i)
                break
            cpt = 0
            while True:
                ltwoout_ = df.sample(n=2)
                u_id = np.unique(ltwoout_['id'].values)
                u_target = np.unique(ltwoout_['target'].values)
                if u_id.size == 2 & u_target.size == 2:
                    break
                cpt += 1
                if cpt > y.size:
                    raise ValueError("Could not build StratifiedLeaveTwoOut folds!")
            df = df.drop(ltwoout_.index)

            testing_samples = ltwoout_
            testing_idx.append(testing_samples.index.tolist())

            tr_idx = np.arange(y.size)
            tr_idx = np.setdiff1d(tr_idx, testing_samples.index.values)

            training_idx.append(tr_idx.tolist())

        for j in range(len(training_idx)):
            yield training_idx[j], testing_idx[j]


if __name__ == "__main__":
    print("***************************")
    print("StratifiedLeaveTwoOut TEST")
    print("***************************")

    class_healthy = 1
    class_unhealthy = 2

    scoring = {
        'balanced_accuracy_score': make_scorer(balanced_accuracy_score),
        # 'roc_auc_score': make_scorer(roc_auc_score, average='weighted'),
        'precision_score0': make_scorer(precision_score, average=None, labels=[class_healthy]),
        'precision_score1': make_scorer(precision_score, average=None, labels=[class_unhealthy]),
        'recall_score0': make_scorer(recall_score, average=None, labels=[class_healthy]),
        'recall_score1': make_scorer(recall_score, average=None, labels=[class_unhealthy]),
        'f1_score0': make_scorer(f1_score, average=None, labels=[class_healthy]),
        'f1_score1': make_scorer(f1_score, average=None, labels=[class_unhealthy])
    }

    X = np.array([[1, 2],
                  [3, 4],
                  [5, 6],
                  [7, 8],
                  [9, 10],
                  [11, 12],
                  [13, 14],
                  [15, 16],
                  [17, 18],
                  [19, 20]])
    animal_ids = np.array(["animal1",
                           "animal1",
                           "animal2",
                           "animal2",
                           "animal3",
                           "animal3",
                           "animal4",
                           "animal4",
                           "animal5",
                           "animal5"])
    y = np.array([1, 1, 2, 2, 2, 1, 1, 2, 2, 1])

    slto = StratifiedLeaveTwoOut(animal_ids, n_repeats=10)

    for train_index, test_index in slto.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print("TRAIN:", train_index, "TEST:", test_index, "TEST TARGET:", y_test, "TEST ANIMAL ID:", animal_ids[test_index])

    print("******************************")
    print("Test custom cv on test dataset")
    print("******************************")
    testdataset = datasets.load_diabetes()
    X = testdataset.data[:, :]
    y = testdataset.target
    #make dataset binary
    m = np.median(y)
    y[y <= m] = 0
    y[y > m] = 1
    y = y.astype(int)

    results = []
    for c in [100, 1.0, 0.1e-05, 0.1e-10, 0.1e-50]:
        clf_std_svc = make_pipeline(SVC(C=c, probability=True, class_weight='balanced'))
        cv_std_svc = StratifiedLeaveTwoOut(y, n_repeats=1000)
        scores = cross_validate(clf_std_svc, X.copy(), y.copy(), cv=cv_std_svc, scoring=scoring, n_jobs=-1)

        df_score = pd.DataFrame(scores)
        acc = np.mean(df_score["test_balanced_accuracy_score"].values)
        results.append(acc)
    print("mean accuracy=", results)


