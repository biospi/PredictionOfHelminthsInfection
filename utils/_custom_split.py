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
    def __init__(self, animal_ids, sample_idx, stratified=False):
        self.nfold = 0
        self.stratified = stratified
        self.sample_idx = np.array(sample_idx).flatten()
        self.animal_ids = np.array(animal_ids).flatten()

    def split(self, X, y, group=None):
        df = pd.DataFrame(np.hstack((y.reshape(y.size, 1), self.animal_ids.reshape(self.animal_ids.size, 1), self.sample_idx.reshape(self.sample_idx.size, 1))))
        #df.to_csv("F:/Data2/test.csv")
        # df = pd.read_csv("F:/Data2/test.csv", index_col=False)
        df = df.apply(pd.to_numeric, downcast='integer')
        df.columns = ["target", "animal_id", "sample_idx"]
        ##df.index = df["sample_idx"]

        groupby_target = pd.DataFrame(df.groupby('animal_id')['target'].apply(list))
        groupby_target["animal_id"] = groupby_target.index
        groupby_sample = pd.DataFrame(df.groupby('animal_id')['sample_idx'].apply(list))
        groupby_sample["animal_id"] = groupby_sample.index

        df_ = groupby_target.copy()
        df_["sample_idx"] = groupby_sample["sample_idx"]
        df_ = df_[['animal_id', 'target', 'sample_idx']]

        print("DATASET:")
        print(df_)
        df_ = df_.values
        training_idx = []
        testing_idx = []
        len_check = []
        for i in range(df_.shape[0]):
            train_idx = []
            test_idx = []
            a1 = df_[i][0]
            if i < df_.shape[0]-1:
                a2 = df_[i + 1][0]
            else:
                a2 = a1
            for j in range(df_.shape[0]):
                a3 = df_[j][0]

                if a3 in [a1, a2]:
                    test_idx.append(df_[j][2])
                else:
                    train_idx.append(df_[j][2])

            # train_idx = np.array(sum(train_idx, [])).flatten()
            # test_idx = np.array(sum(test_idx, [])).flatten()
            all_train_idx = sum(train_idx, [])
            all_test_idx = sum(test_idx, [])
            if self.stratified:
                temp = []
                for e in test_idx:
                    temp.append(df[df['sample_idx'].isin(e)]["target"].tolist())

                s1 = np.unique(np.array(temp[0]))

                if len(temp) == 2:
                    s2 = np.unique(np.array(temp[1]))
                    if s1.size != 1 and s2.size != 1:
                        #samples for the 2 left out animals are not the same target
                        continue
                    s = np.array([s1[0], s2[0]])
                    if np.unique(s).size != 2: #need 1 healthy and 1 unhealthy
                        continue
                else:
                    if s1.size != 1:
                        continue
                    s = np.array(s1[0])
                    if np.unique(s).size != 1:
                        continue

            all_train_idx = df[df['sample_idx'].isin(all_train_idx)].index.tolist()
            all_test_idx = df[df['sample_idx'].isin(all_test_idx)].index.tolist()
            training_idx.append(all_train_idx)
            testing_idx.append(all_test_idx)
            len_check.append(len(test_idx))

            print("FOLD %d --> \nSAMPLE TRAIN IDX:" % i, np.array(all_train_idx), "\nSAMPLE TEST IDX:", np.array(all_test_idx), "\nTEST TARGET:",
                  np.unique(y[all_test_idx]), "\nTEST ANIMAL ID:", np.unique(self.animal_ids[all_test_idx]), "\nTRAIN ANIMAL ID:",
                  np.unique(self.animal_ids[all_train_idx]))

        len_check = np.array(len_check)
        if len_check[len_check > 2].size > 0:
            raise ValueError("fold contains more than 2 testing sample!")

        self.nfold = len(training_idx)
        print("StratifiedLeaveTwoOut could build %d unique folds. stratification=%s" % (self.nfold, self.stratified))
        for n in range(len(training_idx)):
            yield np.array(training_idx[n]), np.array(testing_idx[n])


if __name__ == "__main__":
    print("***************************")
    print("CUSTOM SPLIT TEST")
    print("***************************")

    class_healthy = 1
    class_unhealthy = 4

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


    # y = np.array([4, 4, 1, 1, 4, 4, 1, 1, 1, 1, 4, 4, 1, 1, 4, 4, 1, 4, 4, 4, 1, 4,
    #    1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 1, 4, 1, 4, 1, 1, 1, 4,
    #    4, 4, 1, 1, 1, 1, 4, 4, 1, 4, 4, 4, 4, 1, 4, 4, 4, 1, 1, 1, 1, 1,
    #    1, 1, 1, 4, 4, 4, 4, 1, 1, 1, 4, 1, 1, 1, 1, 4, 4, 4, 1, 4, 4, 1,
    #    4, 4, 4, 1, 4, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 1, 4,
    #    4, 4, 4, 1, 1, 1, 1, 4, 1, 1, 1, 4, 4, 4, 4, 4, 1, 1]).reshape(-1, 1)

    y = np.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]).reshape(-1, 1)

    X = np.array(list(range(y.size))).reshape(-1, 1)

    animal_ids = np.array(['40101310109.0', '40101310109.0', '40101310109.0', '40101310109.0',
       '40101310109.0', '40101310109.0', '40101310109.0', '40101310109.0',
       '40101310109.0', '40101310109.0', '40101310013.0', '40101310013.0',
       '40101310013.0', '40101310013.0', '40101310013.0', '40101310013.0',
       '40101310013.0', '40101310013.0', '40101310134.0', '40101310134.0',
       '40101310134.0', '40101310134.0', '40101310134.0', '40101310134.0',
       '40101310134.0', '40101310134.0', '40101310134.0', '40101310134.0',
       '40101310134.0', '40101310143.0', '40101310143.0', '40101310143.0',
       '40101310143.0', '40101310143.0', '40101310143.0', '40101310143.0',
       '40101310143.0', '40101310143.0', '40101310143.0', '40101310249.0',
       '40101310249.0', '40101310249.0', '40101310249.0', '40101310314.0',
       '40101310314.0', '40101310314.0', '40101310314.0', '40101310314.0',
       '40101310314.0', '40101310314.0', '40101310314.0', '40101310314.0',
       '40101310314.0', '40101310314.0', '40101310314.0', '40101310314.0',
       '40101310316.0', '40101310316.0', '40101310316.0', '40101310316.0',
       '40101310316.0', '40101310316.0', '40101310316.0', '40101310316.0',
       '40101310316.0', '40101310316.0', '40101310316.0', '40101310316.0',
       '40101310316.0', '40101310342.0', '40101310342.0', '40101310342.0',
       '40101310342.0', '40101310342.0', '40101310342.0', '40101310342.0',
       '40101310342.0', '40101310342.0', '40101310342.0', '40101310342.0',
       '40101310342.0', '40101310350.0', '40101310350.0', '40101310350.0',
       '40101310350.0', '40101310350.0', '40101310350.0', '40101310350.0',
       '40101310350.0', '40101310353.0', '40101310353.0', '40101310353.0',
       '40101310353.0', '40101310353.0', '40101310353.0', '40101310353.0',
       '40101310353.0', '40101310353.0', '40101310353.0', '40101310353.0',
       '40101310353.0', '40101310386.0', '40101310386.0', '40101310386.0',
       '40101310386.0', '40101310386.0', '40101310386.0', '40101310386.0',
       '40101310386.0', '40101310386.0', '40101310069.0', '40101310069.0',
       '40101310069.0', '40101310069.0', '40101310069.0', '40101310069.0',
       '40101310069.0', '40101310069.0', '40101310098.0', '40101310098.0',
       '40101310098.0', '40101310098.0', '40101310098.0', '40101310098.0',
       '40101310098.0', '40101310098.0', '40101310098.0', '40101310098.0'])

    sample_idx = [1, 2, 3, 6, 8, 9, 10, 13, 14, 17, 18, 19, 21, 22, 24, 28, 30, 34, 35, 36, 38, 39, 43, 45, 46, 47, 50,
                  51, 52, 55, 56, 59, 60, 61, 62, 63, 65, 67, 69, 71, 73, 76, 80, 81, 82, 85, 87, 88, 89, 90, 91, 92,
                  94, 96, 97, 98, 101, 103, 105, 107, 108, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
                  124, 127, 128, 130, 132, 133, 134, 135, 137, 138, 143, 145, 146, 147, 149, 153, 155, 156, 158, 160,
                  162, 163, 165, 167, 169, 170, 171, 172, 174, 175, 177, 178, 179, 180, 181, 183, 185, 188, 189, 191,
                  194, 197, 198, 199, 201, 205, 206, 207, 209, 210, 211, 214, 215, 219, 220]

    print("DATASET:")
    dataset = pd.DataFrame(np.hstack((X.reshape(X.size, 1), y.reshape(y.size, 1), animal_ids.reshape(animal_ids.size, 1))))
    header = ["activity", "target", "animal_id"]
    dataset.columns = header
    dataset.to_csv("dummy_dataset_for_cv.csv")
    print(dataset)
    print("")

    slto = StratifiedLeaveTwoOut(animal_ids, sample_idx, stratified=True)

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
        #cv_std_svc = StratifiedLeaveTwoOut(animal_ids, sample_idx, stratified=F)
        scores = cross_validate(clf_std_svc, X.copy(), y.copy(), cv=slto, scoring=scoring, n_jobs=-1)

        df_score = pd.DataFrame(scores)
        acc = np.mean(df_score["test_balanced_accuracy_score"].values)
        results.append(acc)
    print("mean accuracy=", results)


