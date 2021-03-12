import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import LeavePOut, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score, balanced_accuracy_score, precision_score, f1_score
from sklearn.metrics import auc

class StratifiedLeaveTwoOut:
    def __init__(self, n_repeats=1):
        self.p = 2
        self.n_repeats = n_repeats

    def split(self, X, y, group=None):
        for idx in range(self.n_repeats):
            cv = LeavePOut(p=self.p)
            for rx, tx in cv.split(X, y):
                s = y[tx]
                u = np.unique(y)
                assert (u.size == 2), "target must be binary!"
                if u[0] in s and u[1] in s:
                    print("TRAIN IDX:", rx, "TEST IDX:", tx, "TARGET:", s)
                    yield rx, tx


if __name__ == "__main__":
    print("start...")

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

    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([1, 1, 2, 2, 2])

    print("X:", X)
    print("y:", y)

    slto = StratifiedLeaveTwoOut(n_repeats=1)

    for train_index, test_index in slto.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # print("TRAIN:", train_index, "TEST:", test_index, y_test)

    clf_std_svc = make_pipeline(preprocessing.StandardScaler(), SVC(probability=True, class_weight='balanced'))
    cv_std_svc = StratifiedLeaveTwoOut()
    scores = cross_validate(clf_std_svc, X.copy(), y.copy(), cv=StratifiedLeaveTwoOut(), scoring=scoring, n_jobs=-1)

    print(scores)