import os
import pathlib
import time
from multiprocessing import Manager, Pool
from pathlib import Path
from sys import exit

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    plot_roc_curve,
    auc,
    roc_curve,
    precision_recall_curve,
    classification_report,
)
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    RepeatedKFold,
    LeaveOneOut,
    GridSearchCV,
)
from sklearn.svm import SVC

#from utils._custom_split import StratifiedLeaveTwoOut
from utils.visualisation import (
    plot_roc_range,
    plot_pr_range,
    plotHeatmap,
    plot_2D_decision_boundaries,
    plot_3D_decision_boundaries,
    build_proba_hist,
)


def downsampleDf(data_frame, class_healthy, class_unhealthy):
    df_true = data_frame[data_frame["target"] == class_unhealthy]
    df_false = data_frame[data_frame["target"] == class_healthy]
    try:
        if df_false.shape[0] > df_true.shape[0]:
            df_false = df_false.sample(df_true.shape[0])
        else:
            print("more 12 to 11.")
            df_true = df_true.sample(df_false.shape[0])
    except ValueError as e:
        print(e)
        return
    data_frame = pd.concat([df_true, df_false], ignore_index=True, sort=False)
    return data_frame


def make_roc_curve(
    class_healthy,
    class_unhealthy,
    clf_name,
    out_dir,
    classifier,
    X,
    y,
    cv,
    steps,
    cv_name,
    animal_ids,
    days,
    split1=None,
    split2=None,
    tag="",
):
    steps = clf_name + "_" + steps
    print("make_roc_curve %s" % cv_name, steps)
    if isinstance(X, pd.DataFrame):
        X = X.values

    if isinstance(cv, LeaveOneOut):
        roc_auc = LeaveOnOutRoc(
            classifier, X, y, out_dir, cv_name, steps, animal_ids, cv, days
        )
        return roc_auc
    else:
        y_ground_truth_pr = []
        y_proba_pr = []
        tprs = []
        aucs_roc = []
        aucs_pr = []
        precisions = []
        recalls = []
        mean_fpr = np.linspace(0, 1, 100)
        plt.clf()
        fig_roc, ax_roc = plt.subplots(figsize=(8.00, 6.00))
        fig_pr, ax_pr = plt.subplots(figsize=(8.00, 6.00))
        y_binary = (y.copy() != 1).astype(int)

        data0 = []
        data1 = []

        if cv is None:
            if split1 is None:
                a = [
                    (np.arange(0, int(y.size / 2)), np.arange(int(y.size / 2), y.size))
                ]  # split in half
            else:
                a = [(np.arange(0, split1), np.arange(split1, y.size))]
        else:
            a = cv.split(X, y)

        for i, (train, test) in enumerate(a):
            classifier.fit(X[train], y[train])
            y_proba_test = classifier.predict_proba(X[test])[:, 1]
            y_bin_test = y_binary[test]
            h_0 = y_proba_test[y_bin_test == 0]
            h_1 = y_proba_test[y_bin_test == 1]
            data0.extend(h_0)
            data1.extend(h_1)
            if isinstance(cv, StratifiedLeaveTwoOut):
                print("make_roc_curve fold %d/%d" % (i, cv.nfold))
                viz_roc = plot_roc_curve(classifier, X[test], y[test])
                # viz_pr = plot_precision_recall_curve(classifier, X[test], y_binary[test])

                label = "%d auc=%d idx=%d" % (
                    int(float(np.unique(cv.animal_ids[test])[0])),
                    viz_roc.roc_auc * 100,
                    test[0],
                )
                if viz_roc.roc_auc > 0.95:
                    viz_roc = plot_roc_curve(
                        classifier,
                        X[test],
                        y[test],
                        label=label,
                        alpha=1,
                        lw=1.5,
                        ax=ax_roc,
                    )
                    precision, recall, _ = precision_recall_curve(
                        y_bin_test, y_proba_test
                    )
                    ax_pr.step(recall, precision, label=label, lw=1.5)
                elif viz_roc.roc_auc < 0.2:
                    viz_roc = plot_roc_curve(
                        classifier,
                        X[test],
                        y[test],
                        label=label,
                        alpha=1,
                        lw=1.5,
                        ax=ax_roc,
                    )
                    precision, recall, _ = precision_recall_curve(
                        y_bin_test, y_proba_test
                    )
                    ax_pr.step(recall, precision, label=label, lw=1.5)
                else:
                    viz_roc = plot_roc_curve(
                        classifier,
                        X[test],
                        y[test],
                        label=None,
                        alpha=0.3,
                        lw=1,
                        ax=ax_roc,
                        c="tab:blue",
                    )
                    precision, recall, _ = precision_recall_curve(
                        y_bin_test, y_proba_test
                    )
                    ax_pr.step(recall, precision, label=None)
            else:
                if cv is not None:
                    print(
                        "make_roc_curve fold %d/%d"
                        % (i, cv.n_repeats * cv.cvargs["n_splits"])
                    )
                else:
                    print(f"make_roc_curve split={i}")

                if animal_ids is not None:
                    animal_ids = np.array(animal_ids)
                    print(
                        "FOLD %d --> \nSAMPLE TRAIN IDX:" % i,
                        train,
                        "\nSAMPLE TEST IDX:",
                        test,
                        "\nTEST TARGET:",
                        np.unique(y[test]),
                        "\nTRAIN TARGET:",
                        np.unique(y[train]),
                        "\nTEST ANIMAL ID:",
                        np.unique(animal_ids[test]),
                        "\nTRAIN ANIMAL ID:",
                        np.unique(animal_ids[train]),
                    )

                viz_roc = plot_roc_curve(
                    classifier,
                    X[test],
                    y[test],
                    label=None,
                    alpha=0.3,
                    lw=1,
                    ax=ax_roc,
                    c="tab:blue",
                )
                precision, recall, _ = precision_recall_curve(y_bin_test, y_proba_test)
                y_pred = classifier.predict(X[test])
                ax_pr.step(recall, precision, label=None, lw=1, c="tab:blue")

            interp_tpr = np.interp(mean_fpr, viz_roc.fpr, viz_roc.tpr)
            interp_tpr[0] = 0.0
            print("auc=", viz_roc.roc_auc)
            if "TSNE(2)" in steps or "UMAP" in steps:
                plot_2D_decision_boundaries(
                    viz_roc.roc_auc,
                    i,
                    X,
                    y,
                    X[test],
                    y[test],
                    X[train],
                    y[train],
                    steps,
                    classifier,
                    out_dir,
                    steps,
                    DR="TSNE",
                )
            if "PCA(3)" in steps and "linear" in steps.lower():
                plot_3D_decision_boundaries(
                    X,
                    y,
                    X[train],
                    y[train],
                    X[test],
                    y[test],
                    steps,
                    classifier,
                    i,
                    out_dir,
                    steps,
                    viz_roc.roc_auc,
                )
            if "TSNE(3)" in steps and "linear" in steps.lower():
                plot_3D_decision_boundaries(
                    X,
                    y,
                    X[train],
                    y[train],
                    X[test],
                    y[test],
                    steps,
                    classifier,
                    i,
                    out_dir,
                    steps,
                    viz_roc.roc_auc,
                    DR="TSNE",
                )

            if np.isnan(viz_roc.roc_auc):
                continue
            tprs.append(interp_tpr)
            aucs_roc.append(viz_roc.roc_auc)
            aucs_pr.append(auc(recall, precision))
            precisions.append(precision)
            info = f"train shape:{str(train.shape)} healthy:{np.sum(y[train] == class_healthy)} unhealthy:{np.sum(y[train] == class_unhealthy)}| test shape:{str(test.shape)} healthy:{np.sum(y[test] == class_healthy)} unhealthy:{np.sum(y[test] == class_unhealthy)}"
            recalls.append(recall)

            y_ground_truth_pr.append(y_binary[test])
            y_proba_pr.append(classifier.predict_proba(X[test])[:, 1])

            # ax.plot(viz.fpr, viz.tpr, c="tab:green")

        print("make_roc_curve done!")
        mean_auc = plot_roc_range(
            ax_roc,
            tprs,
            mean_fpr,
            aucs_roc,
            out_dir,
            steps,
            fig_roc,
            cv_name,
            days,
            info=info,
            tag=tag,
        )
        mean_auc_pr = plot_pr_range(
            ax_pr,
            y_ground_truth_pr,
            y_proba_pr,
            aucs_pr,
            out_dir,
            steps,
            fig_pr,
            cv_name,
            days,
        )

        plt.close(fig_roc)
        plt.close(fig_pr)
        plt.clf()
        makeYHist(data0, data1, out_dir, cv_name, steps, mean_auc, info=info, tag=tag)
        return mean_auc, aucs_roc


def process_data_frame_svm(
    N_META,
    output_dir,
    animal_ids,
    data_frame,
    days,
    farm_id,
    steps,
    n_splits,
    n_repeats,
    sampling,
    downsample_false_class,
    label_series,
    class_healthy,
    class_unhealthy,
    class_healthy_label,
    class_unhealthy_label,
    y_col="target",
    cv=None,
    n_job=6,
):
    print("*******************************************************************")
    mlp_layers = (1000, 500, 100, 45, 30, 15)
    print(label_series)
    data_frame["id"] = animal_ids
    # data_frame = data_frame.loc[
    #     data_frame["target"].isin([class_healthy, class_unhealthy])
    # ]
    if downsample_false_class:
        data_frame = downsampleDf(data_frame, class_healthy, class_unhealthy)

    # animal_ids = data_frame["id"].tolist()
    sample_idxs = data_frame.index.tolist()

    # if cv == "StratifiedLeaveTwoOut":
    #     cross_validation_method = StratifiedLeaveTwoOut(
    #         animal_ids, sample_idxs, stratified=True, verbose=True
    #     )
    #
    # if cv == "LeaveTwoOut":
    #     cross_validation_method = StratifiedLeaveTwoOut(
    #         animal_ids, sample_idxs, stratified=False, verbose=True
    #     )

    if cv == "RepeatedStratifiedKFold":
        cross_validation_method = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=0
        )

    if cv == "RepeatedKFold":
        cross_validation_method = RepeatedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=0
        )

    if cv == "RepeatedKFoldMRNN":
        from utils._custom_split import RepeatedKFoldCustom
        cross_validation_method = RepeatedKFoldCustom(
            2,
            2,
            metadata=None,
            days=days,
            farmname="delmas",
            method="MRNN",
            out_dir=output_dir,
            N_META=N_META,
            full_activity_data_file=Path(
                "C:/Users/fo18103/PycharmProjects/PredictionOfHelminthsInfection/Data/delmas_activity_data_weather.csv"
            ),
        )

    if cv == "LeaveOneOut":
        cross_validation_method = LeaveOneOut()

    data_frame = data_frame.drop("id", 1)

    y = data_frame[y_col].values.flatten()
    y = y.astype(int)
    X = data_frame[data_frame.columns[0 : data_frame.shape[1] - 1]].values

    print("release data_frame memory...")
    del data_frame
    print("****************************")

    if not os.path.exists(output_dir):
        print("mkdir", output_dir)
        os.makedirs(output_dir)

    plotHeatmap(
        X,
        output_dir,
        "CLF_INPUT_%s" % steps,
        "CLF_INPUT_%s.html" % steps,
        xaxis="features",
        yaxis="value",
    )
    # plotAllFeatures(X, y, output_dir, filename="CLF_ALLFEATURES_%s.html" % steps)

    # filename_2d_scatter = (
    #     "%s/PLS/%s_2DPLS_days_%d_option_%s_downsampled_%s_sampling_%s.png"
    #     % (output_dir, farm_id, days, steps, downsample_false_class, sampling)
    # )

    # pls = PLSRegression(n_components=2)
    # X_pls = pls.fit_transform(X.copy(), y.copy())[0]
    # plot_2d_space(
    #     X_pls, y, filename_2d_scatter, label_series, "2 PLS components " + steps
    # )

    print("************************************************")
    print("downsample on= " + str(downsample_false_class))
    class0_count = str(y[y == class_healthy].size)
    class1_count = str(y[y == class_unhealthy].size)
    print("X-> class0=" + class0_count + " class1=" + class1_count)
    # try:
    #     if int(class1_count) < 2 or int(class0_count) < 2:
    #         print("not enough samples!")
    #         return
    # except ValueError as e:
    #     print(e)
    #     return
    #
    # classes_all = np.unique(y)
    #
    # scoring = {
    #     "balanced_accuracy_score": make_scorer(balanced_accuracy_score),
    #     # 'roc_auc_score': make_scorer(roc_auc_score, average='weighted'),
    #     "precision_score0": make_scorer(
    #         precision_score, average=None, labels=[class_healthy]
    #     ),
    #     "precision_score1": make_scorer(
    #         precision_score, average=None, labels=[class_unhealthy]
    #     ),
    #     "recall_score0": make_scorer(
    #         recall_score, average=None, labels=[class_healthy]
    #     ),
    #     "recall_score1": make_scorer(
    #         recall_score, average=None, labels=[class_unhealthy]
    #     ),
    #     "f1_score0": make_scorer(f1_score, average=None, labels=[class_healthy]),
    #     "f1_score1": make_scorer(f1_score, average=None, labels=[class_unhealthy]),
    # }

    # for c_a in classes_all:
    #     scoring[f"precision_score{c_a}"] = make_scorer(precision_score, average=None, labels=[c_a])
    #     scoring[f"recall_score{c_a}"] = make_scorer(recall_score, average=None, labels=[c_a])
    #     scoring[f"f1_score{c_a}"] = make_scorer(f1_score, average=None, labels=[c_a])

    # param_str = "option_%s_downsample_%s_days_%d_farmid_%s_nrepeat_%d_nsplits_%d_class0_%s_class1_%s_sampling_%s" % (
    #     steps, str(downsample_false_class), days, farm_id, n_repeats, n_splits, class0_count,
    #     class1_count, sampling)
    report_rows_list = []

    scores = cross_validate_custom_fast(
        output_dir,
        steps,
        cv,
        days,
        label_series,
        class_healthy,
        class_unhealthy,
        cross_validation_method,
        X,
        y,
        n_job,
    )

    # scores = cross_validate_custom(
    #     output_dir,
    #     steps,
    #     cv,
    #     days,
    #     label_series,
    #     class_healthy,
    #     class_unhealthy,
    #     cross_validation_method,
    #     X,
    #     y,
    # )

    build_proba_hist(output_dir, class_unhealthy_label, scores)

    exit()

    for clf_svc in [
        SVC(kernel="linear", probability=True, class_weight="balanced"),
        SVC(kernel="rbf", probability=True, class_weight="balanced"),
    ]:

        # tuned_parameters = [{'kernel': ['rbf'], 'gamma': ['scale', 1e-1, 1e-3, 1e-4], 'class_weight': [None, 'balanced'],
        #                      'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]},
        #                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
        # clf = GridSearchCV(clf_svc, tuned_parameters, cv=cross_validation_method, scoring='roc_auc', n_jobs=-1)
        # clf.fit(X.copy(), y.copy())
        # clf_best = clf.best_estimator_
        # print("Best estimator from gridsearch=")
        # print(clf_best)
        # scores = cross_validate(
        #     clf_svc,
        #     X.copy(),
        #     y.copy(),
        #     cv=cross_validation_method,
        #     scoring=scoring,
        #     n_jobs=-1,
        #     return_estimator=True,
        # )

        scores["downsample"] = downsample_false_class
        scores["class0"] = y[y == class_healthy].size
        scores["class1"] = y[y == class_unhealthy].size
        scores["steps"] = steps
        scores["days"] = days
        scores["farm_id"] = farm_id
        scores["balanced_accuracy_score_mean"] = np.mean(
            scores["test_balanced_accuracy_score"]
        )
        scores["precision_score0_mean"] = np.mean(scores["test_precision_score0"])
        scores["precision_score1_mean"] = np.mean(scores["test_precision_score1"])
        scores["recall_score0_mean"] = np.mean(scores["test_recall_score0"])
        scores["recall_score1_mean"] = np.mean(scores["test_recall_score1"])
        scores["f1_score0_mean"] = np.mean(scores["test_f1_score0"])
        scores["f1_score1_mean"] = np.mean(scores["test_f1_score1"])
        scores["sampling"] = sampling
        scores["classifier"] = "->SVC(%s)" % clf_svc.kernel
        scores["classifier_details"] = str(clf_svc).replace("\n", "").replace(" ", "")

        # clf_svc = make_pipeline(SVC(probability=True, class_weight='balanced'))
        auc_m, aucs = make_roc_curve(
            class_healthy,
            class_unhealthy,
            scores["classifier"].replace("->", ""),
            output_dir,
            clf_svc,
            X.copy(),
            y.copy(),
            cross_validation_method,
            steps,
            cv,
            animal_ids,
            days,
        )
        scores["roc_auc_score_mean"] = auc_m
        scores["roc_auc_scores"] = aucs
        report_rows_list.append(scores)

        df_report = pd.DataFrame(report_rows_list)
        df_report["class_0_label"] = str(class_healthy_label)
        df_report["class_1_label"] = str(class_unhealthy_label)
        df_report["nfold"] = (
            cross_validation_method.nfold
            if hasattr(cross_validation_method, "nfold")
            else np.nan
        )
        # df_report["n_splits"] = cross_validation_method.cvargs['n_splits'] if hasattr(cross_validation_method,
        #                                                                               'cvargs') else np.nan
        # df_report["n_repeats"] = cross_validation_method.n_repeats if hasattr(cross_validation_method,
        #                                                                       'n_repeats') else np.nan
        df_report["total_fit_time"] = [
            time.strftime("%H:%M:%S", time.gmtime(np.nansum(x)))
            for x in df_report["fit_time"].values
        ]
        # filename = "%s/%s/%s_%s_classification_report_days_%d_option_%s_downsampled_%s_sampling_%s.csv" % (
        #     output_dir, cv, scores["classifier"].replace("->", ""), farm_id, days, steps, downsample_false_class, sampling)

        out = output_dir / cv
        out.mkdir(parents=True, exist_ok=True)
        filename = (
            out
            / f"{scores['classifier'].replace('->', '')}_{farm_id}_classification_report_days_{days}_{steps}_downsampled_{downsample_false_class}_sampling_{sampling}.csv"
        )
        # create_rec_dir(filename)
        df_report.to_csv(filename, sep=",", index=False)
        print("filename=", filename)
        del scores

    # model_files = []
    # for clf_fitted in [
    #     SVC(kernel="linear", probability=True, class_weight="balanced"),
    #     SVC(kernel="rbf", probability=True, class_weight="balanced"),
    # ]:
    #     clf_fitted = clf_fitted.fit(X.copy(), y.copy())
    #     filename = output_dir / cv / f"model_{days}_{steps}_{clf_fitted.kernel}.pkl"
    #     model_files.append(filename)
    #     print("saving classifier...")
    #     print(filename)
    #     with open(str(filename), "wb") as f:
    #         pickle.dump(clf_fitted, f)
    # return model_files


def cross_validate_custom(
    out_dir,
    steps,
    cv_name,
    days,
    label_series,
    class_healthy,
    class_unhealthy,
    cross_validation_method,
    X,
    y,
):
    """Cross validate X,y data and plot roc curve with range
    Args:
        out_dir: output directory to save figures to
        steps: postprocessing steps
        cv_name: name of cross validation method
        days: count of activity days in sample
        label_series: dict that holds famacha label/target
        class_healthy: target integer of healthy class
        class_unhealthy: target integer of unhealthy class
        cross_validation_method: Cv object
        X: samples
        y: targets
    """
    scores = {}
    for clf in [
        SVC(kernel="linear", probability=True, class_weight="balanced"),
        SVC(kernel="rbf", probability=True, class_weight="balanced"),
    ]:
        plt.clf()
        fig_roc, ax_roc = plt.subplots(figsize=(8.00, 6.00))
        mean_fpr = np.linspace(0, 1, 100)

        fold_results = []
        fold_probas = {}
        for k in label_series.values():
            fold_probas[k] = []

        tprs = []
        aucs_roc = []
        for ifold, (train_index, test_index) in enumerate(
            cross_validation_method.split(X, y)
        ):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # hold all extra label
            fold_index = np.array(train_index.tolist() + test_index.tolist())
            X_fold = X[fold_index]
            y_fold = y[fold_index]

            # keep healthy and unhealthy only
            X_train = X_train[np.isin(y_train, [class_healthy, class_unhealthy])]
            y_train = y_train[np.isin(y_train, [class_healthy, class_unhealthy])]

            X_test = X_test[np.isin(y_test, [class_healthy, class_unhealthy])]
            y_test = y_test[np.isin(y_test, [class_healthy, class_unhealthy])]

            clf.fit(X_train, y_train)

            # test healthy/unhealthy
            y_pred = clf.predict(X_test)
            y_pred_proba = clf.predict_proba(X_test)

            # prep for roc curve
            viz_roc = plot_roc_curve(
                clf,
                X_test,
                y_test,
                label=None,
                alpha=0.3,
                lw=1,
                ax=ax_roc,
                c="tab:blue",
            )

            interp_tpr = np.interp(mean_fpr, viz_roc.fpr, viz_roc.tpr)
            interp_tpr[0] = 0.0
            print("auc=", viz_roc.roc_auc)
            tprs.append(interp_tpr)
            aucs_roc.append(viz_roc.roc_auc)

            accuracy = balanced_accuracy_score(y_test, y_pred)
            precision, recall, fscore, support = precision_recall_fscore_support(
                y_test, y_pred
            )

            fold_result = {
                "target": class_unhealthy,
                "accuracy": accuracy,
                "class_healthy": class_healthy,
                "class_unhealthy": class_unhealthy,
                "y_test": y_test,
                "test_precision_score_0": precision[0],
                "test_precision_score_1": precision[1],
                "test_recall_0": recall[0],
                "test_recall_1": recall[1],
                "test_fscore_0": fscore[0],
                "test_fscore_1": fscore[1],
                "test_support_0": support[0],
                "test_support_1": support[1],
            }
            fold_results.append(fold_result)

            # test individual labels and store probabilities to be healthy/unhealthy
            for y_f in y_fold:
                label = label_series[y_f]
                X_test = X_fold[y_fold == y_f]
                y_test = y_fold[y_fold == y_f]
                y_pred_proba = clf.predict_proba(X_test)
                fold_proba = {
                    "test_y_pred_proba_0": y_pred_proba[:, 0],
                    "test_y_pred_proba_1": y_pred_proba[:, 1],
                }
                fold_probas[label].append(fold_proba)

        info = f"X shape:{str(X.shape)} healthy:{np.sum(y == class_healthy)} unhealthy:{np.sum(y == class_unhealthy)}"
        mean_auc = plot_roc_range(
            ax_roc,
            tprs,
            mean_fpr,
            aucs_roc,
            out_dir,
            steps,
            fig_roc,
            cv_name,
            days,
            info=info,
            tag=f"{type(clf).__name__}_{clf.kernel}",
        )

        scores[f"{type(clf).__name__}_{clf.kernel}_results"] = fold_results
        scores[f"{type(clf).__name__}_{clf.kernel}_probas"] = fold_probas

    return scores


def fold_worker(
    tprs,
    aucs_roc,
    fold_results,
    fold_probas,
    label_series,
    mean_fpr,
    clf,
    X,
    y,
    train_index,
    test_index,
    class_healthy,
    class_unhealthy,
    axis,
    ifold,
    nfold
):
    print(f"process id={ifold}/{nfold}...")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # hold all extra label
    fold_index = np.array(train_index.tolist() + test_index.tolist())
    X_fold = X[fold_index]
    y_fold = y[fold_index]

    # keep healthy and unhealthy only
    X_train = X_train[np.isin(y_train, [class_healthy, class_unhealthy])]
    y_train = y_train[np.isin(y_train, [class_healthy, class_unhealthy])]

    X_test = X_test[np.isin(y_test, [class_healthy, class_unhealthy])]
    y_test = y_test[np.isin(y_test, [class_healthy, class_unhealthy])]

    clf.fit(X_train, y_train)

    # test healthy/unhealthy
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)

    # prep for roc curve
    viz_roc = plot_roc_curve(
        clf,
        X_test,
        y_test,
        label=None,
        alpha=0.3,
        lw=1,
        ax=None,
        c="tab:blue",
    )
    axis.append(viz_roc)

    interp_tpr = np.interp(mean_fpr, viz_roc.fpr, viz_roc.tpr)
    interp_tpr[0] = 0.0
    print("auc=", viz_roc.roc_auc)
    tprs.append(interp_tpr)
    aucs_roc.append(viz_roc.roc_auc)

    accuracy = balanced_accuracy_score(y_test, y_pred)
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred)

    fold_result = {
        "target": class_unhealthy,
        "accuracy": accuracy,
        "class_healthy": class_healthy,
        "class_unhealthy": class_unhealthy,
        "y_test": y_test,
        "test_precision_score_0": precision[0],
        "test_precision_score_1": precision[1],
        "test_recall_0": recall[0],
        "test_recall_1": recall[1],
        "test_fscore_0": fscore[0],
        "test_fscore_1": fscore[1],
        "test_support_0": support[0],
        "test_support_1": support[1],
    }
    fold_results.append(fold_result)

    # test individual labels and store probabilities to be healthy/unhealthy
    for y_f in y_fold:
        label = label_series[y_f]
        X_test = X_fold[y_fold == y_f]
        y_test = y_fold[y_fold == y_f]
        y_pred_proba = clf.predict_proba(X_test)
        fold_proba = {
            "test_y_pred_proba_0": y_pred_proba[:, 0],
            "test_y_pred_proba_1": y_pred_proba[:, 1],
        }
        fold_probas[label].append(fold_proba)


def cross_validate_custom_fast(
    out_dir,
    steps,
    cv_name,
    days,
    label_series,
    class_healthy,
    class_unhealthy,
    cross_validation_method,
    X,
    y,
    n_job=None,
):
    """Cross validate X,y data and plot roc curve with range
    Args:
        out_dir: output directory to save figures to
        steps: postprocessing steps
        cv_name: name of cross validation method
        days: count of activity days in sample
        label_series: dict that holds famacha label/target
        class_healthy: target integer of healthy class
        class_unhealthy: target integer of unhealthy class
        cross_validation_method: Cv object
        X: samples
        y: targets
    """
    scores = {}
    for clf in [
        SVC(kernel="linear", probability=True, class_weight="balanced"),
        SVC(kernel="rbf", probability=True, class_weight="balanced"),
    ]:
        plt.clf()
        fig_roc, ax_roc = plt.subplots(figsize=(8.00, 6.00))
        mean_fpr = np.linspace(0, 1, 100)

        with Manager() as manager:
            #create result holders
            tprs = manager.list()
            axis = manager.list()
            aucs_roc = manager.list()
            fold_results = manager.list()
            fold_probas = manager.dict()
            for k in label_series.values():
                fold_probas[k] = manager.list()

            pool = Pool(processes=n_job)
            start = time.time()
            for ifold, (train_index, test_index) in enumerate(
                cross_validation_method.split(X, y)
            ):
                pool.apply_async(fold_worker, (
                    tprs,
                    aucs_roc,
                    fold_results,
                    fold_probas,
                    label_series,
                    mean_fpr,
                    clf,
                    X,
                    y,
                    train_index,
                    test_index,
                    class_healthy,
                    class_unhealthy,
                    axis,
                    ifold,
                    cross_validation_method.get_n_splits()
                ))
            pool.close()
            pool.join()
            end = time.time()
            fold_results = list(fold_results)
            axis = list(axis)
            tprs = list(tprs)
            aucs_roc = list(aucs_roc)
            fold_probas = dict(fold_probas)
            fold_probas = dict([a, list(x)] for a, x in fold_probas.items())
            print('total time (s)= ' + str(end - start))

        info = f"X shape:{str(X.shape)} healthy:{np.sum(y == class_healthy)} unhealthy:{np.sum(y == class_unhealthy)}"
        for a in axis:
            f, ax = a.figure_, a.ax_
            xdata = ax.lines[0].get_xdata()
            ydata = ax.lines[0].get_ydata()
            ax_roc.plot(xdata, ydata, color='tab:blue', alpha=.3, linewidth=1)

        mean_auc = plot_roc_range(
            ax_roc,
            tprs,
            mean_fpr,
            aucs_roc,
            out_dir,
            steps,
            fig_roc,
            cv_name,
            days,
            info=info,
            tag=f"{type(clf).__name__}_{clf.kernel}",
        )

        scores[f"{type(clf).__name__}_{clf.kernel}_results"] = fold_results
        scores[f"{type(clf).__name__}_{clf.kernel}_probas"] = fold_probas

    return scores


def LeaveOnOutRoc(clf, X, y, out_dir, cv_name, classifier_name, animal_ids, cv, days):
    all_y = []
    all_probs = []
    i = 0
    y_binary = (y.copy() != 1).astype(int)
    n = cv.get_n_splits(X, y_binary)
    for train, test in cv.split(X, y_binary):
        animal_ids = np.array(animal_ids)
        print("make_roc_curve fold %d/%d" % (i, n))
        print(
            "FOLD %d --> \nSAMPLE TRAIN IDX:" % i,
            train,
            "\nSAMPLE TEST IDX:",
            test,
            "\nTEST TARGET:",
            np.unique(y_binary[test]),
            "\nTRAIN TARGET:",
            np.unique(y_binary[train]),
            "\nTEST ANIMAL ID:",
            np.unique(animal_ids[test]),
            "\nTRAIN ANIMAL ID:",
            np.unique(animal_ids[train]),
        )
        i += 1
        all_y.append(y_binary[test])

        y_predict_proba = clf.fit(X[train], y_binary[train]).predict_proba(X[test])[
            :, 1
        ]
        all_probs.append(y_predict_proba)

    all_y = np.array(all_y)
    all_probs = np.array(all_probs)

    fpr, tpr, thresholds = roc_curve(all_y, all_probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(12.80, 7.20))
    ax.plot(fpr, tpr, lw=2, alpha=0.5, label="LOOCV ROC (AUC = %0.2f)" % (roc_auc))
    ax.plot(
        [0, 1], [0, 1], linestyle="--", lw=2, color="k", label="Chance level", alpha=0.8
    )
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(
        "Receiver operating characteristic LOOCV (at sample level) days=%d" % days
    )
    ax.legend(loc="lower right")
    ax.grid()
    path = out_dir / "roc_curve" / cv_name
    final_path = path / f"roc_{classifier_name}.png"
    print(final_path)
    fig.savefig(final_path)
    plt.close(fig)
    plt.clf()

    precision, recall, thresholds = precision_recall_curve(all_y, all_probs)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots(figsize=(12.80, 7.20))
    ax.plot(
        recall, precision, lw=2, alpha=0.5, label="LOOCV PR (AUC = %0.2f)" % (pr_auc)
    )
    # ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Chance level', alpha=.8)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision Recall LOOCV (at sample level)")
    ax.legend(loc="lower right")
    ax.grid()
    path = out_dir / "pr_curve" / cv_name
    final_path = path / f"pr_{classifier_name}.png"
    print(final_path)
    fig.savefig(final_path)
    plt.close(fig)
    plt.clf()
    return roc_auc


def makeYHist(data0, data1, out_dir, cv_name, steps, auc, info="", tag=""):
    plt.figure(figsize=(8, 6))
    plt.hist(data0, bins=100, alpha=0.5, label="healthy (y=0)")
    plt.hist(data1, bins=100, alpha=0.5, label="unhealthy (y=1)")
    plt.xlabel("Y probability", size=14)
    plt.ylabel("Count", size=14)
    plt.title(
        "Histograms of prediction probabilities (ROC Mean AUC = %0.2f)\n %s"
        % (auc, info)
    )
    plt.legend(loc="upper right")
    filename = "%s/%s_overlapping_histograms_y_%s_%s.png" % (
        out_dir,
        tag,
        cv_name,
        steps,
    )
    print(filename)
    plt.savefig(filename)
    plt.clf()
    # plt.show()


# def process_data_frame_svm(
#     meta,
#     N_META,
#     mrnn_files,
#     output_dir,
#     animal_ids,
#     data_frame,
#     days,
#     farm_id,
#     steps,
#     n_splits,
#     n_repeats,
#     sampling,
#     downsample_false_class,
#     label_series,
#     class_healthy,
#     class_unhealthy,
#     class_healthy_label,
#     class_unhealthy_label,
#     y_col="target",
#     cv=None,
# ):
#     print("*******************************************************************")
#     mlp_layers = (1000, 500, 100, 45, 30, 15)
#     print(label_series)
#     data_frame["id"] = animal_ids
#     data_frame = data_frame.loc[
#         data_frame["target"].isin([class_healthy, class_unhealthy])
#     ]
#     if downsample_false_class:
#         data_frame = downsampleDf(data_frame, class_healthy, class_unhealthy)
#
#     # animal_ids = data_frame["id"].tolist()
#     sample_idxs = data_frame.index.tolist()
#
#     if cv == "StratifiedLeaveTwoOut":
#         cross_validation_method = StratifiedLeaveTwoOut(
#             animal_ids, sample_idxs, stratified=True, verbose=True
#         )
#
#     if cv == "LeaveTwoOut":
#         cross_validation_method = StratifiedLeaveTwoOut(
#             animal_ids, sample_idxs, stratified=False, verbose=True
#         )
#
#     if cv == "RepeatedStratifiedKFold":
#         cross_validation_method = RepeatedStratifiedKFold(
#             n_splits=n_splits, n_repeats=n_repeats, random_state=0
#         )
#
#     if cv == "RepeatedKFold":
#         cross_validation_method = RepeatedKFold(
#             n_splits=n_splits, n_repeats=n_repeats, random_state=0
#         )
#
#     if cv == "RepeatedKFoldMRNN":
#         cross_validation_method = RepeatedKFoldCustom(
#             2,
#             2,
#             mrnn_samples=mrnn_files,
#             metadata=meta,
#             days=days,
#             farmname="delmas",
#             method="MRNN",
#             out_dir=output_dir,
#             N_META=N_META,
#             full_activity_data_file=Path(
#                 "C:/Users/fo18103/PycharmProjects/PredictionOfHelminthsInfection/Data/delmas_activity_data_weather.csv"
#             )
#         )
#
#     if cv == "LeaveOneOut":
#         cross_validation_method = LeaveOneOut()
#
#     data_frame = data_frame.drop("id", 1)
#
#     y = data_frame[y_col].values.flatten()
#     y = y.astype(int)
#     X = data_frame[data_frame.columns[0 : data_frame.shape[1] - 1]].values
#
#     print("release data_frame memory...")
#     del data_frame
#     print("****************************")
#
#     if not os.path.exists(output_dir):
#         print("mkdir", output_dir)
#         os.makedirs(output_dir)
#
#     plotHeatmap(
#         X,
#         output_dir,
#         "CLF_INPUT_%s" % steps,
#         "CLF_INPUT_%s.html" % steps,
#         xaxis="features",
#         yaxis="value",
#     )
#     # plotAllFeatures(X, y, output_dir, filename="CLF_ALLFEATURES_%s.html" % steps)
#
#     # filename_2d_scatter = (
#     #     "%s/PLS/%s_2DPLS_days_%d_option_%s_downsampled_%s_sampling_%s.png"
#     #     % (output_dir, farm_id, days, steps, downsample_false_class, sampling)
#     # )
#
#     # pls = PLSRegression(n_components=2)
#     # X_pls = pls.fit_transform(X.copy(), y.copy())[0]
#     # plot_2d_space(
#     #     X_pls, y, filename_2d_scatter, label_series, "2 PLS components " + steps
#     # )
#
#     print("************************************************")
#     print("downsample on= " + str(downsample_false_class))
#     class0_count = str(y[y == class_healthy].size)
#     class1_count = str(y[y == class_unhealthy].size)
#     print("X-> class0=" + class0_count + " class1=" + class1_count)
#     try:
#         if int(class1_count) < 2 or int(class0_count) < 2:
#             print("not enough samples!")
#             return
#     except ValueError as e:
#         print(e)
#         return
#
#     scoring = {
#         "balanced_accuracy_score": make_scorer(balanced_accuracy_score),
#         # 'roc_auc_score': make_scorer(roc_auc_score, average='weighted'),
#         "precision_score0": make_scorer(
#             precision_score, average=None, labels=[class_healthy]
#         ),
#         "precision_score1": make_scorer(
#             precision_score, average=None, labels=[class_unhealthy]
#         ),
#         "recall_score0": make_scorer(
#             recall_score, average=None, labels=[class_healthy]
#         ),
#         "recall_score1": make_scorer(
#             recall_score, average=None, labels=[class_unhealthy]
#         ),
#         "f1_score0": make_scorer(f1_score, average=None, labels=[class_healthy]),
#         "f1_score1": make_scorer(f1_score, average=None, labels=[class_unhealthy]),
#     }
#
#     # param_str = "option_%s_downsample_%s_days_%d_farmid_%s_nrepeat_%d_nsplits_%d_class0_%s_class1_%s_sampling_%s" % (
#     #     steps, str(downsample_false_class), days, farm_id, n_repeats, n_splits, class0_count,
#     #     class1_count, sampling)
#     report_rows_list = []
#
#     for clf_svc in [
#         SVC(kernel="linear", probability=True, class_weight="balanced"),
#         SVC(kernel="rbf", probability=True, class_weight="balanced"),
#     ]:
#         # tuned_parameters = [{'kernel': ['rbf'], 'gamma': ['scale', 1e-1, 1e-3, 1e-4], 'class_weight': [None, 'balanced'],
#         #                      'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]},
#         #                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
#         # clf = GridSearchCV(clf_svc, tuned_parameters, cv=cross_validation_method, scoring='roc_auc', n_jobs=-1)
#         # clf.fit(X.copy(), y.copy())
#         # clf_best = clf.best_estimator_
#         # print("Best estimator from gridsearch=")
#         # print(clf_best)
#         scores = cross_validate(
#             clf_svc,
#             X.copy(),
#             y.copy(),
#             cv=cross_validation_method,
#             scoring=scoring,
#             n_jobs=-1,
#             return_estimator=True,
#         )
#         scores["downsample"] = downsample_false_class
#         scores["class0"] = y[y == class_healthy].size
#         scores["class1"] = y[y == class_unhealthy].size
#         scores["steps"] = steps
#         scores["days"] = days
#         scores["farm_id"] = farm_id
#         scores["balanced_accuracy_score_mean"] = np.mean(
#             scores["test_balanced_accuracy_score"]
#         )
#         scores["precision_score0_mean"] = np.mean(scores["test_precision_score0"])
#         scores["precision_score1_mean"] = np.mean(scores["test_precision_score1"])
#         scores["recall_score0_mean"] = np.mean(scores["test_recall_score0"])
#         scores["recall_score1_mean"] = np.mean(scores["test_recall_score1"])
#         scores["f1_score0_mean"] = np.mean(scores["test_f1_score0"])
#         scores["f1_score1_mean"] = np.mean(scores["test_f1_score1"])
#         scores["sampling"] = sampling
#         scores["classifier"] = "->SVC(%s)" % clf_svc.kernel
#         scores["classifier_details"] = str(clf_svc).replace("\n", "").replace(" ", "")
#         # clf_svc = make_pipeline(SVC(probability=True, class_weight='balanced'))
#         auc_m, aucs = make_roc_curve(
#             class_healthy,
#             class_unhealthy,
#             scores["classifier"].replace("->", ""),
#             output_dir,
#             clf_svc,
#             X.copy(),
#             y.copy(),
#             cross_validation_method,
#             steps,
#             cv,
#             animal_ids,
#             days,
#         )
#         scores["roc_auc_score_mean"] = auc_m
#         scores["roc_auc_scores"] = aucs
#         report_rows_list.append(scores)
#
#         df_report = pd.DataFrame(report_rows_list)
#         df_report["class_0_label"] = str(class_healthy_label)
#         df_report["class_1_label"] = str(class_unhealthy_label)
#         df_report["nfold"] = (
#             cross_validation_method.nfold
#             if hasattr(cross_validation_method, "nfold")
#             else np.nan
#         )
#         # df_report["n_splits"] = cross_validation_method.cvargs['n_splits'] if hasattr(cross_validation_method,
#         #                                                                               'cvargs') else np.nan
#         # df_report["n_repeats"] = cross_validation_method.n_repeats if hasattr(cross_validation_method,
#         #                                                                       'n_repeats') else np.nan
#         df_report["total_fit_time"] = [
#             time.strftime("%H:%M:%S", time.gmtime(np.nansum(x)))
#             for x in df_report["fit_time"].values
#         ]
#         # filename = "%s/%s/%s_%s_classification_report_days_%d_option_%s_downsampled_%s_sampling_%s.csv" % (
#         #     output_dir, cv, scores["classifier"].replace("->", ""), farm_id, days, steps, downsample_false_class, sampling)
#
#         out = output_dir / cv
#         out.mkdir(parents=True, exist_ok=True)
#         filename = (
#             out
#             / f"{scores['classifier'].replace('->', '')}_{farm_id}_classification_report_days_{days}_{steps}_downsampled_{downsample_false_class}_sampling_{sampling}.csv"
#         )
#         # create_rec_dir(filename)
#         df_report.to_csv(filename, sep=",", index=False)
#         print("filename=", filename)
#         del scores
#
#     model_files = []
#     for clf_fitted in [
#         SVC(kernel="linear", probability=True, class_weight="balanced"),
#         SVC(kernel="rbf", probability=True, class_weight="balanced"),
#     ]:
#         clf_fitted = clf_fitted.fit(X.copy(), y.copy())
#         filename = output_dir / cv / f"model_{days}_{steps}_{clf_fitted.kernel}.pkl"
#         model_files.append(filename)
#         print("saving classifier...")
#         print(filename)
#         with open(str(filename), "wb") as f:
#             pickle.dump(clf_fitted, f)
#     return model_files


def processSVM(X_train, X_test, y_train, y_test, output_dir):
    clf_svc = SVC(kernel="rbf", probability=True, class_weight="balanced")
    tuned_parameters = [
        {
            "kernel": ["rbf"],
            "gamma": ["scale", 1e-1, 1e-3, 1e-4],
            "class_weight": [None, "balanced"],
            "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
        },
        {"kernel": ["linear"], "C": [1, 10, 100, 1000]},
    ]

    clf = GridSearchCV(
        clf_svc,
        tuned_parameters,
        scoring=["roc_auc", "accuracy", "precision"],
        refit="accuracy",
        n_jobs=-1,
    )
    clf.fit(X_train.copy(), y_train.copy())
    clf_best = clf.best_estimator_
    print("Best estimator from gridsearch=")
    print(clf_best)
    y_pred = clf.predict(X_test.copy())
    print(classification_report(y_test, y_pred))
    print(f"precision_score: {precision_score(y_test, y_pred, average='weighted')}")
    filename = "%s/report.csv" % output_dir
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(filename)
    df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True))
    df.to_csv(filename)
    X = np.array(X_train.tolist() + X_test.tolist())
    y = np.array(y_train.tolist() + y_test.tolist())
    return clf_best, X, y
