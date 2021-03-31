import warnings

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score, balanced_accuracy_score, precision_score, f1_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sys import exit
from sklearn.model_selection import LeavePOut

np.random.seed(0)


class StratifiedLeaveTwoOut:
    def __init__(self, animal_ids, stratified=False):
        self.nfold = 0
        self.stratified = stratified
        self.animal_ids = np.array(animal_ids).flatten()

    def split(self, X, y, group=None):
        df = pd.DataFrame(np.hstack((y.reshape(y.size, 1), self.animal_ids.reshape(self.animal_ids.size, 1))))
        df.columns = ["target", "animal_id"]
        df = pd.DataFrame(df.groupby('animal_id')['target'].apply(list))
        df.reset_index(level=0, inplace=True)
        idx = 0
        idxs = []
        for index, row in df.iterrows():
            temp = []
            for _ in row['target']:
                temp.append(idx)
                idx += 1
            idxs.append(temp)
        df["s_indexes"] = idxs
        print("DATASET:")
        print(df)
        df = df.values
        training_idx = []
        testing_idx = []

        for i in range(df.shape[0]):
            train_idx = []
            test_idx = []
            a1 = df[i][0]
            if i < df.shape[0]-1:
                a2 = df[i + 1][0]
            else:
                a2 = a1
            for j in range(df.shape[0]):
                a3 = df[j][0]

                if a3 in [a1, a2]:
                    test_idx.append(df[j][2])
                else:
                    train_idx.append(df[j][2])

            train_idx = np.array(sum(train_idx, []))
            test_idx = np.array(sum(test_idx, [])).flatten()

            if self.stratified:
                if np.unique(y[test_idx]).shape[0] != 2:
                    continue

            training_idx.append(train_idx)
            testing_idx.append(test_idx)
            print("FOLD %d --> SAMPLE TRAIN IDX:" % i, train_idx, "SAMPLE TEST IDX:", test_idx, "TEST TARGET:",
                  y[test_idx], "TEST ANIMAL ID:", np.unique(self.animal_ids[test_idx]), "TRAIN ANIMAL ID:",
                  np.unique(self.animal_ids[train_idx]))

        self.nfold = len(training_idx)
        print("StratifiedLeaveTwoOut could build %d unique folds. stratification=%s" % (self.nfold, self.stratified))
        for n in range(len(training_idx)):
            yield np.array(training_idx[n]), np.array(testing_idx[n])


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
                  [7, 8],
                  [9, 10],
                  [7, 8],
                  [9, 10],
                  [11, 12]])
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
    y = np.array([2,
                 1,
                 2,
                 2,
                 2,
                 2,
                 2,
                 2,
                 1,
                 1])

    # y = np.array(["healthy",
    #              "healthy",
    #              "uhealthy",
    #              "uhealthy",
    #               "uhealthy",
    #               "uhealthy",
    #               "uhealthy",
    #               "uhealthy",
    #              "healthy",
    #              "healthy"])
    print("DATASET:")
    dataset = pd.DataFrame(np.hstack((X, y.reshape(y.size, 1), animal_ids.reshape(animal_ids.size, 1))))
    header = ["a_%d" % x for x in range(X.shape[1])] + ["target", "animal_id"]
    dataset.columns = header
    dataset.to_csv("dummy_dataset_for_cv.csv")
    print(dataset)
    print("")

    slto = StratifiedLeaveTwoOut(animal_ids)


    rows = []
    i = 0
    for train_index, test_index in slto.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #print("FOLD %d --> SAMPLE TRAIN IDX:" % i, train_index, "SAMPLE TEST IDX:", test_index, "TEST TARGET:", y_test, "TEST ANIMAL ID:", np.unique(animal_ids[test_index]), "TRAIN ANIMAL ID:", np.unique(animal_ids[train_index]))
        row = train_index.tolist() + test_index.tolist() + y_test.tolist() + animal_ids[test_index].tolist()
        rows.append(row)
        i += 1
    # print(slto.nfold)
    # df_rows = pd.DataFrame(rows)
    # df_rows.columns = [ ["TRAIN IDX" for x in range(len(train_index))] + ["TEST IDX" for x in range(len(test_index))] + ["TEST TARGET" for x in range(len(y_test))] + ["TEST ANIMAL ID" for x in range(len(animal_ids[test_index].tolist()))] ]
    # df_rows.to_csv("stratified_leave_two_out_folds.csv")
    # print("******************************")
    # print("Test custom cv on test dataset")
    # print("******************************")
    # testdataset = datasets.load_diabetes()
    # X = testdataset.data[:, :]
    # y = testdataset.target
    # #make dataset binary
    # m = np.median(y)
    # y[y <= m] = 0
    # y[y > m] = 1
    # y = y.astype(int)

    results = []
    for c in [100]:
        clf_std_svc = make_pipeline(SVC(C=c, probability=True, class_weight='balanced'))
        cv_std_svc = StratifiedLeaveTwoOut(animal_ids, stratified=True)
        scores = cross_validate(clf_std_svc, X.copy(), y.copy(), cv=cv_std_svc, scoring=scoring, n_jobs=-1)

        df_score = pd.DataFrame(scores)
        acc = np.mean(df_score["test_balanced_accuracy_score"].values)
        results.append(acc)
    print("mean accuracy=", results)


