import warnings

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score, balanced_accuracy_score, precision_score, f1_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

np.random.seed(0)


class StratifiedLeaveTwoOut:
    def __init__(self, animal_ids, n_repeats=10, stratified=True):
        self.n_repeats = n_repeats
        self.stratified = stratified
        self.animal_ids = np.array(animal_ids).flatten()

    def split(self, X, y, verbose=True, group=None):
        y = y.reshape(y.size, 1)
        animal_ids = self.animal_ids.reshape(self.animal_ids.size, 1)
        hstack = np.hstack((X, y, animal_ids))
        df = pd.DataFrame(hstack)
        header = ["f_" + str(x) for x in df.columns.tolist()]
        header[-1] = "id"
        header[-2] = "target"
        df.columns = header

        training_idx = []
        testing_idx = []
        iter = 0
        if verbose:
            print(df)
        while True:
            cpt = 0
            while True:
                ltwoout_ = df.sample(n=2)
                u_id = np.unique(ltwoout_['id'].values)
                u_target = np.unique(ltwoout_['target'].values)
                if self.stratified:
                    if u_id.size == 2 & u_target.size == 2:
                        break
                else:
                    if u_id.size == 2:
                        break
                cpt += 1
                if cpt > y.size:
                    warnings.warn("Could not build StratifiedLeaveTwoOut folds!")
                    break

            testing_samples = ltwoout_
            tr_idx = np.arange(y.size)
            tr_idx = np.setdiff1d(tr_idx, testing_samples.index.values)
            df_train = pd.DataFrame(training_idx)
            df_test = pd.DataFrame(testing_idx)
            unique = len(df_test[~df_test.apply(frozenset, axis=1).duplicated()]) #drop row permutably
            # unique = len(df_test.drop_duplicates())

            # d = df_test.drop_duplicates()
            d = df_test[~df_test.apply(frozenset, axis=1).duplicated()]
            testing_idx = d.values.tolist()

            training_idx = df_train[df_train.index.isin(d.index)].values.tolist()
            testing_idx.append(testing_samples.index.tolist())
            training_idx.append(tr_idx.tolist())

            if len(training_idx) >= self.n_repeats:
                break

            if unique == self.n_repeats:
                break

            if iter > self.n_repeats * 100:
                warnings.warn("cannot build more folds MAX REPEAT=%d" % iter)
                break
            iter += 1

        training_idx = training_idx[0:self.n_repeats]
        print("StratifiedLeaveTwoOut could build %d/%d unique folds." % (len(training_idx), self.n_repeats))
        for j in range(len(training_idx)):
            yield training_idx[j], testing_idx[j]


if __name__ == "__main__":
    print("***************************")
    print("CUSTOM SPLIT TEST")
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
    # y = np.array([1,
    #              1,
    #              2,
    #              2,
    #              1,
    #              1,
    #              2,
    #              2,
    #              1,
    #              1])

    y = np.array(["healthy",
                 "healthy",
                 "uhealthy",
                 "uhealthy",
                 "healthy",
                 "healthy",
                 "uhealthy",
                 "uhealthy",
                 "healthy",
                 "healthy"])

    dataset = pd.DataFrame(np.hstack((X, y.reshape(y.size, 1), animal_ids.reshape(animal_ids.size, 1))))
    dataset.columns = [["f_%d" % x for x in range(X.shape[1])] + ["target", "animal_id"]]
    dataset.to_csv("dummy_dataset_for_cv.csv")

    slto = StratifiedLeaveTwoOut(animal_ids, n_repeats=10, stratified=True)

    rows = []
    for train_index, test_index in slto.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print("TRAIN IDX:", train_index, "TEST IDX:", test_index, "TEST TARGET:", y_test, "TEST ANIMAL ID:", animal_ids[test_index].tolist())
        row = train_index + test_index + y_test.tolist() + animal_ids[test_index].tolist()
        rows.append(row)

    df_rows = pd.DataFrame(rows)
    df_rows.columns = [ ["TRAIN IDX" for x in range(len(train_index))] + ["TEST IDX" for x in range(len(test_index))] + ["TEST TARGET" for x in range(len(y_test))] + ["TEST ANIMAL ID" for x in range(len(animal_ids[test_index].tolist()))] ]
    df_rows.to_csv("stratified_leave_two_out_folds.csv")
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


