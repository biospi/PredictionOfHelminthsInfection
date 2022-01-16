import warnings

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score, balanced_accuracy_score, precision_score, f1_score
from sklearn.model_selection import cross_validate, RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sys import exit
from sklearn.model_selection import LeavePOut
import itertools
from pathlib import Path
from model.data_loader import load_activity_data
from mrnn.main_mrnn import start_mrnn
from utils._anscombe import Anscombe
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime
import matplotlib.pyplot as plt
np.random.seed(0)


def plot_heatmap(
    X,
    timestamps,
    animal_ids,
    out_dir,
    title="Heatmap",
    filename="heatmap.html",
    yaxis="Count",
    xaxis="Time bin(1D)",
):
    fig = make_subplots(rows=1, cols=1)
    trace = go.Heatmap(
        z=X.T,
        x=timestamps,
        y=animal_ids,
        colorscale="Viridis",
    )
    fig.add_trace(trace, row=1, col=1)
    fig.update_layout(title_text=title)
    fig.update_layout(xaxis_title=xaxis)
    fig.update_layout(yaxis_title=yaxis)
    # fig.show()
    # create_rec_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / filename.replace("=", "_").lower()
    print(file_path)
    fig.write_html(str(file_path))
    return trace, title


def remove_testing_samples(activity_file, out_dir, ifold, df, to_remove, nmin=10080):
    print("removing testing samples from dataset....")
    df_without_test = df.copy()
    df_without_test["date_datetime"] = pd.to_datetime(df_without_test["date_str"])
    for test_sample in to_remove:
        animal_id = int(test_sample[0])
        date = pd.to_datetime(test_sample[2], format="%d/%m/%Y")
        date_range = pd.date_range(end=date, periods=nmin, freq='T')
        #grab test sample from full dataset and remove activity values
        df_without_test.loc[df_without_test[[str(animal_id), 'date_datetime']]['date_datetime'].isin(date_range), str(animal_id)] = -1
    df_resampled = df_without_test.resample('D', on='date_datetime').sum()
    filtered_columns = [x for x in df_resampled.columns if x.isdigit()]
    df_resampled = df_resampled[filtered_columns]
    df_resampled[df_resampled._get_numeric_data() < 0] = np.nan
    timestamps = [pd.to_datetime(x) for x in df_resampled.index.values]
    plot_heatmap(df_resampled.values, timestamps, [f"_{x}" for x in filtered_columns],
                 title=f"Fold {ifold} training data with testing samples removed",
                 out_dir=out_dir / Path("CV"),
                 filename=f"fold_{ifold}.html")
    print("removed testing data. ready for imputation.")
    filename = f"{activity_file.stem}_{ifold}.csv"
    out = out_dir / Path("CV") / filename
    print(out)
    df_without_test[df_without_test._get_numeric_data() < 0] = np.nan
    df_without_test.to_csv(out)
    return df_without_test, out, filtered_columns


class RepeatedKFoldCustom:
    def __init__(self, n_splits, n_repeats, random_state=0, farmname='', out_dir = None, metadata = None,
                 full_activity_data_file = None, days = None, method = None):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.farmname = farmname
        self.out_dir = out_dir
        self.metadata = metadata
        self.full_activity_data_file = full_activity_data_file
        self.days = days
        self.method = method

    def split(self, X, y=None, groups=None):
        print('imputing...')
        df = pd.read_csv(self.full_activity_data_file)
        #X_i, y_i = [], []
        rkf = RepeatedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.random_state)
        for ifold, (train_index, test_index) in enumerate(rkf.split(X, y)):
            X_train_meta, X_test_meta = self.metadata[train_index], self.metadata[test_index]
            df, file_without_test, transponders = remove_testing_samples(self.full_activity_data_file, out_dir, ifold, df, X_test_meta)
            if self.method == "MRNN":
                model, norm_parameters, streams = start_mrnn(self.farmname, file_without_test)
            if self.method == "GAIN":
                model, norm_parameters, streams = start_mrnn(self.farmname, file_without_test)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            X_train = model.transform_(X_train, X_train_meta, streams, self.days, norm_parameters)
            X_test = model.transform_(X_test, X_test_meta, streams, self.days, norm_parameters)
            X[train_index] = X_train
            X[test_index] = X_test
            yield train_index, test_index

        # print("mrrn imputation done!")
        # path = Path(os.path.dirname(os.path.dirname(__file__)))
        # print(f"set current dir to {path}")

    # def split(self, X, y):
    #     rkf = RepeatedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.random_state)
    #     i = 0
    #     for train_index, test_index in rkf.split(X):
    #         print(f"FOLD {i} --> TRAIN:{train_index} TEST:{test_index}")
    #         i += 1
    #         yield train_index, test_index


class StratifiedLeaveTwoOut:
    def __init__(self, animal_ids, sample_idx, stratified=False, verbose=False):
        self.nfold = 0
        self.verbose = verbose
        self.stratified = stratified
        self.sample_idx = np.array(sample_idx).flatten()
        self.animal_ids = np.array(animal_ids).flatten()

    def split(self, X, y, group=None, leaven=2):
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

        if self.verbose:
            print("DATASET:")
            print(df_)

        a = df_["animal_id"].tolist()
        comb = []
        for i in range(0, len(a) + 1):
            for subset in itertools.combinations(a, i):
                if len(subset) != leaven:
                    continue
                if subset not in comb:
                    comb.append(subset)
        comb = np.array(comb)

        training_idx = []
        testing_idx = []
        len_check = []
        map = dict(df["sample_idx"])
        map = dict(zip(map.values(), map.keys()))
        for i, c in enumerate(comb):
            test_idx = df_[df_["animal_id"].isin(c)]["sample_idx"].tolist()
            all_test_idx = sum(test_idx, [])
            all_test_idx = [map[x] for x in all_test_idx]
            train_idx = df_[~df_["animal_id"].isin(c)]["sample_idx"].tolist()
            all_train_idx = sum(train_idx, [])
            all_train_idx = [map[x] for x in all_train_idx]

            if self.stratified:
                temp = []
                for e in test_idx:
                    temp.append(df[df['sample_idx'].isin(e)]["target"].tolist())

                s1 = np.unique(np.array(temp[0]))

                if len(temp) == leaven:
                    s2 = np.unique(np.array(temp[1]))
                    if s1.size != 1 and s2.size != 1:
                        #samples for the 2 left out animals are not the same target
                        continue
                    s = np.array([s1[0], s2[0]])
                    if np.unique(s).size != leaven: #need 1 healthy and 1 unhealthy
                        continue
                else:
                    if s1.size != 1:
                        continue
                    s = np.array(s1[0])
                    if np.unique(s).size != 1:
                        continue
            if np.unique(y[all_train_idx]).size == 1:
                warnings.warn("Cannot use fold for training! Only 1 target in FOLD %d" % i)
                continue

            training_idx.append(all_train_idx)
            testing_idx.append(all_test_idx)
            len_check.append(len(test_idx))
            if self.verbose:
                print("FOLD %d --> \nSAMPLE TRAIN IDX:" % i, np.array(all_train_idx), "\nSAMPLE TEST IDX:", np.array(all_test_idx), "\nTEST TARGET:",
                      np.unique(y[all_test_idx]), "\nTRAIN TARGET:",
                      np.unique(y[all_train_idx]), "\nTEST ANIMAL ID:", np.unique(self.animal_ids[all_test_idx]), "\nTRAIN ANIMAL ID:",
                      np.unique(self.animal_ids[all_train_idx]))

        len_check = np.array(len_check)
        if len_check[len_check > leaven].size > 0:
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
    #
    #
    # # y = np.array([4, 4, 1, 1, 4, 4, 1, 1, 1, 1, 4, 4, 1, 1, 4, 4, 1, 4, 4, 4, 1, 4,
    # #    1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 1, 4, 1, 4, 1, 1, 1, 4,
    # #    4, 4, 1, 1, 1, 1, 4, 4, 1, 4, 4, 4, 4, 1, 4, 4, 4, 1, 1, 1, 1, 1,
    # #    1, 1, 1, 4, 4, 4, 4, 1, 1, 1, 4, 1, 1, 1, 1, 4, 4, 4, 1, 4, 4, 1,
    # #    4, 4, 4, 1, 4, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 1, 4,
    # #    4, 4, 4, 1, 1, 1, 1, 4, 1, 1, 1, 4, 4, 4, 4, 4, 1, 1]).reshape(-1, 1)
    #
    # y = np.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #    1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]).reshape(-1, 1)
    #
    # X = np.array(list(range(y.size))).reshape(-1, 1)
    # X = np.concatenate((X, X), 1)

    file = Path("E:/Data2/debug3/delmas/datasetraw_none_7day/activity_farmid_dbft_7_1min.csv")
    out_dir = Path("E:/Data2/cv_test")
    day = 7
    imputed_days = 7
    print(f"loading dataset file {file} ...")
    (
        data_frame,
        N_META,
        class_healthy_target,
        class_unhealthy_target,
        label_series,
        samples,
    ) = load_activity_data(file, day, ["1To1"], ["2To2"])

    X = data_frame.iloc[:, :-N_META].values
    y = data_frame["target"].values


    print(f"loading dataset file {file} ...")
    (
        data_frame_,
        _,
        _,
        _,
        _,
        _,
    ) = load_activity_data(file, day, ["1To1"], ["2To2"], preprocessing_steps=[['LINEAR']])

    X_ = data_frame_.iloc[:, :-N_META].values
    y_ = data_frame_["target"].values

    rkf = RepeatedKFold(2, 2)
    clf_std_svc = make_pipeline(SVC(probability=True, class_weight='balanced'))
    # cv_std_svc = StratifiedLeaveTwoOut(animal_ids, sample_idx, stratified=F)
    scores = cross_validate(clf_std_svc, X_.copy(), y_.copy(), cv=rkf, scoring=scoring, n_jobs=-1)

    df_score = pd.DataFrame(scores)
    acc = np.mean(df_score["test_balanced_accuracy_score"].values)
    print(df_score)
    print(acc)

    meta = data_frame.iloc[:, data_frame.shape[1]-N_META:].values

    rkfc = RepeatedKFoldCustom(2, 2, farmname="delmas",
                               metadata=meta,
                               days=day,
                               method="MRNN",
                               out_dir=out_dir,
                               full_activity_data_file=Path("C:/Users/fo18103/PycharmProjects/PredictionOfHelminthsInfection/Data/activity_data.csv"))

    clf_std_svc = make_pipeline(SVC(probability=True, class_weight='balanced'))
    # cv_std_svc = StratifiedLeaveTwoOut(animal_ids, sample_idx, stratified=F)
    scores = cross_validate(clf_std_svc, X.copy(), y.copy(), cv=rkfc, scoring=scoring, n_jobs=-1)

    df_score = pd.DataFrame(scores)
    acc = np.mean(df_score["test_balanced_accuracy_score"].values)
    print(df_score)
    print(acc)
    exit()

    #X, y = rkfc.split(X, y)
    # for train_index, test_index in rkfc.split(X, y):
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]

    #exit()

    # animal_ids = np.array(['40101310109.0', '40101310109.0', '40101310109.0', '40101310109.0',
    #    '40101310109.0', '40101310109.0', '40101310109.0', '40101310109.0',
    #    '40101310109.0', '40101310109.0', '40101310013.0', '40101310013.0',
    #    '40101310013.0', '40101310013.0', '40101310013.0', '40101310013.0',
    #    '40101310013.0', '40101310013.0', '40101310134.0', '40101310134.0',
    #    '40101310134.0', '40101310134.0', '40101310134.0', '40101310134.0',
    #    '40101310134.0', '40101310134.0', '40101310134.0', '40101310134.0',
    #    '40101310134.0', '40101310143.0', '40101310143.0', '40101310143.0',
    #    '40101310143.0', '40101310143.0', '40101310143.0', '40101310143.0',
    #    '40101310143.0', '40101310143.0', '40101310143.0', '40101310249.0',
    #    '40101310249.0', '40101310249.0', '40101310249.0', '40101310314.0',
    #    '40101310314.0', '40101310314.0', '40101310314.0', '40101310314.0',
    #    '40101310314.0', '40101310314.0', '40101310314.0', '40101310314.0',
    #    '40101310314.0', '40101310314.0', '40101310314.0', '40101310314.0',
    #    '40101310316.0', '40101310316.0', '40101310316.0', '40101310316.0',
    #    '40101310316.0', '40101310316.0', '40101310316.0', '40101310316.0',
    #    '40101310316.0', '40101310316.0', '40101310316.0', '40101310316.0',
    #    '40101310316.0', '40101310342.0', '40101310342.0', '40101310342.0',
    #    '40101310342.0', '40101310342.0', '40101310342.0', '40101310342.0',
    #    '40101310342.0', '40101310342.0', '40101310342.0', '40101310342.0',
    #    '40101310342.0', '40101310350.0', '40101310350.0', '40101310350.0',
    #    '40101310350.0', '40101310350.0', '40101310350.0', '40101310350.0',
    #    '40101310350.0', '40101310353.0', '40101310353.0', '40101310353.0',
    #    '40101310353.0', '40101310353.0', '40101310353.0', '40101310353.0',
    #    '40101310353.0', '40101310353.0', '40101310353.0', '40101310353.0',
    #    '40101310353.0', '40101310386.0', '40101310386.0', '40101310386.0',
    #    '40101310386.0', '40101310386.0', '40101310386.0', '40101310386.0',
    #    '40101310386.0', '40101310386.0', '40101310069.0', '40101310069.0',
    #    '40101310069.0', '40101310069.0', '40101310069.0', '40101310069.0',
    #    '40101310069.0', '40101310069.0', '40101310098.0', '40101310098.0',
    #    '40101310098.0', '40101310098.0', '40101310098.0', '40101310098.0',
    #    '40101310098.0', '40101310098.0', '40101310098.0', '40101310098.0'])
    #
    # sample_idx = [1, 2, 3, 6, 8, 9, 10, 13, 14, 17, 18, 19, 21, 22, 24, 28, 30, 34, 35, 36, 38, 39, 43, 45, 46, 47, 50,
    #               51, 52, 55, 56, 59, 60, 61, 62, 63, 65, 67, 69, 71, 73, 76, 80, 81, 82, 85, 87, 88, 89, 90, 91, 92,
    #               94, 96, 97, 98, 101, 103, 105, 107, 108, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
    #               124, 127, 128, 130, 132, 133, 134, 135, 137, 138, 143, 145, 146, 147, 149, 153, 155, 156, 158, 160,
    #               162, 163, 165, 167, 169, 170, 171, 172, 174, 175, 177, 178, 179, 180, 181, 183, 185, 188, 189, 191,
    #               194, 197, 198, 199, 201, 205, 206, 207, 209, 210, 211, 214, 215, 219, 220]
    #
    # print("DATASET:")
    # dataset = pd.DataFrame(np.hstack((X, y.reshape(y.size, 1), animal_ids.reshape(animal_ids.size, 1))))
    # header = ["a1", "a2", "target", "animal_id"]
    # dataset.columns = header
    # dataset.to_csv("dummy_dataset_for_cv.csv")
    # print(dataset)
    # print("")
    #
    # slto = StratifiedLeaveTwoOut(animal_ids, sample_idx, stratified=False, verbose=True)
    #
    # rows = []
    # i = 0
    # for train_index, test_index in slto.split(X, y):
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    #     #print("FOLD %d --> SAMPLE TRAIN IDX:" % i, train_index, "SAMPLE TEST IDX:", test_index, "TEST TARGET:", y_test, "TEST ANIMAL ID:", np.unique(animal_ids[test_index]), "TRAIN ANIMAL ID:", np.unique(animal_ids[train_index]))
    #     row = train_index.tolist() + test_index.tolist() + y_test.tolist() + animal_ids[test_index].tolist()
    #     rows.append(row)
    #     i += 1
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
        scores = cross_validate(clf_std_svc, X.copy(), y.copy(), cv=rkfc, scoring=scoring, n_jobs=-1)

        df_score = pd.DataFrame(scores)
        acc = np.mean(df_score["test_balanced_accuracy_score"].values)
        results.append(acc)
    print("without anscombe mean accuracy=", results)

    results = []
    for c in [100]:
        clf_std_svc = make_pipeline(Anscombe(), SVC(C=c, probability=True, class_weight='balanced'))
        #cv_std_svc = StratifiedLeaveTwoOut(animal_ids, sample_idx, stratified=F)
        scores = cross_validate(clf_std_svc, X.copy(), y.copy(), cv=rkfc, scoring=scoring, n_jobs=-1)

        df_score = pd.DataFrame(scores)
        acc = np.mean(df_score["test_balanced_accuracy_score"].values)
        results.append(acc)
    print("with anscombe mean accuracy=", results)


