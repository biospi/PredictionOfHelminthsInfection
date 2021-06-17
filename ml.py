#
# Author: Axel Montout <axel.montout <a.t> bristol.ac.uk>
#
# Copyright (C) 2020  Biospi Laboratory for Medical Bioinformatics, University of Bristol, UK
#
# This file is part of PredictionOfHelminthsInfection.
#
# PHI is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PHI is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with seaMass.  If not, see <http://www.gnu.org/licenses/>.
#

#%%
import argparse
import gc
import glob
import os
import pathlib
import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer, balanced_accuracy_score, precision_score, recall_score, f1_score, \
    plot_roc_curve, auc, roc_curve, precision_recall_curve, plot_precision_recall_curve
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, LeaveOneOut, GridSearchCV, RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC

from cnn.cnn import run1DCnn, run2DCnn
from utils.Utils import create_rec_dir
from utils._anscombe import Anscombe, Log, Sqrt
from utils._custom_split import StratifiedLeaveTwoOut
from cwt._cwt import CWT, CWTVisualisation, STFT
from utils._normalisation import QuotientNormalizer, CenterScaler, BaseLineScaler
from utils.visualisation import plot_2d_space, plotMlReport, plot_roc_range, plotDistribution, plotMeanGroups, \
    plot_zeros_distrib, plot_groups, plot_time_lda, plot_time_pca, plot_pr_range, SampleVisualisation, plotHeatmap, \
    plot_2D_decision_boundaries, plot_3D_decision_boundaries


def LeaveOnOutRoc(clf, X, y, out_dir, cv_name, classifier_name, animal_ids, cv, days):
    all_y = []
    all_probs = []
    i = 0
    y_binary = (y.copy() != 1).astype(int)
    n = cv.get_n_splits(X, y_binary)
    for train, test in cv.split(X, y_binary):
        animal_ids = np.array(animal_ids)
        print("make_roc_curve fold %d/%d" % (i, n))
        print("FOLD %d --> \nSAMPLE TRAIN IDX:" % i, train, "\nSAMPLE TEST IDX:",
              test, "\nTEST TARGET:",
              np.unique(y_binary[test]), "\nTRAIN TARGET:",
              np.unique(y_binary[train]), "\nTEST ANIMAL ID:", np.unique(animal_ids[test]),
              "\nTRAIN ANIMAL ID:",
              np.unique(animal_ids[train]))
        i += 1
        all_y.append(y_binary[test])

        y_predict_proba = clf.fit(X[train], y_binary[train]).predict_proba(X[test])[:, 1]
        all_probs.append(y_predict_proba)

    all_y = np.array(all_y)
    all_probs = np.array(all_probs)

    fpr, tpr, thresholds = roc_curve(all_y, all_probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(12.80, 7.20))
    ax.plot(fpr, tpr, lw=2, alpha=0.5, label='LOOCV ROC (AUC = %0.2f)' % (roc_auc))
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Chance level', alpha=.8)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic LOOCV (at sample level) days=%d' % days)
    ax.legend(loc="lower right")
    ax.grid()
    path = "%s/roc_curve/%s/" % (out_dir, cv_name)
    create_rec_dir(path)
    final_path = '%s/%s' % (path, 'roc_%s.png' % classifier_name)
    print(final_path)
    fig.savefig(final_path)
    plt.close(fig)
    plt.clf()

    precision, recall, thresholds = precision_recall_curve(all_y, all_probs)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots(figsize=(12.80, 7.20))
    ax.plot(recall, precision, lw=2, alpha=0.5, label='LOOCV PR (AUC = %0.2f)' % (pr_auc))
    #ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Chance level', alpha=.8)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision Recall LOOCV (at sample level)')
    ax.legend(loc="lower right")
    ax.grid()
    path = "%s/pr_curve/%s/" % (out_dir, cv_name)
    create_rec_dir(path)
    final_path = '%s/%s' % (path, 'pr_%s.png' % classifier_name)
    print(final_path)
    fig.savefig(final_path)
    plt.close(fig)
    plt.clf()

    return roc_auc


def makeRocCurve(clf_name, out_dir, classifier, X, y, cv, steps, cv_name, animal_ids, days):
    steps = clf_name +"_"+ steps
    print("make_roc_curve %s" % cv_name)
    if isinstance(X, pd.DataFrame):
        X = X.values

    if isinstance(cv, LeaveOneOut):
        roc_auc = LeaveOnOutRoc(classifier, X, y, out_dir, cv_name, steps, animal_ids, cv, days)
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
        fig_roc, ax_roc = plt.subplots(figsize=(12.80, 7.20))
        fig_pr, ax_pr = plt.subplots(figsize=(12.80, 7.20))
        y_binary = (y.copy() != 1).astype(int)
        for i, (train, test) in enumerate(cv.split(X, y)):
            classifier.fit(X[train], y[train])
            if isinstance(cv, StratifiedLeaveTwoOut):
                print("make_roc_curve fold %d/%d" % (i, cv.nfold))
                viz_roc = plot_roc_curve(classifier, X[test], y[test])
                #viz_pr = plot_precision_recall_curve(classifier, X[test], y_binary[test])

                label = "%d auc=%d idx=%d" % (int(float(np.unique(cv.animal_ids[test])[0])), viz_roc.roc_auc*100, test[0])
                if viz_roc.roc_auc > 0.95:
                    viz_roc = plot_roc_curve(classifier, X[test], y[test],
                                         label=label,
                                         alpha=1, lw=1.5, ax=ax_roc)
                    precision, recall, _ = precision_recall_curve(y_binary[test], classifier.predict_proba(X[test])[:, 1])
                    ax_pr.step(recall, precision, label=label, lw=1.5)
                elif viz_roc.roc_auc < 0.2:
                    viz_roc = plot_roc_curve(classifier, X[test], y[test],
                                         label=label,
                                         alpha=1, lw=1.5, ax=ax_roc)
                    precision, recall, _ = precision_recall_curve(y_binary[test], classifier.predict_proba(X[test])[:, 1])
                    ax_pr.step(recall, precision, label=label, lw=1.5)
                else:
                    viz_roc = plot_roc_curve(classifier, X[test], y[test],
                                         label=None,
                                         alpha=0.3, lw=1, ax=ax_roc, c="tab:blue")
                    precision, recall, _ = precision_recall_curve(y_binary[test], classifier.predict_proba(X[test])[:, 1])
                    ax_pr.step(recall, precision, label=None)
            else:
                print("make_roc_curve fold %d/%d" % (i, cv.n_repeats * cv.cvargs['n_splits']))
                animal_ids = np.array(animal_ids)
                print("FOLD %d --> \nSAMPLE TRAIN IDX:" % i, train, "\nSAMPLE TEST IDX:",
                      test, "\nTEST TARGET:",
                      np.unique(y[test]), "\nTRAIN TARGET:",
                      np.unique(y[train]), "\nTEST ANIMAL ID:", np.unique(animal_ids[test]),
                      "\nTRAIN ANIMAL ID:",
                      np.unique(animal_ids[train]))
                viz_roc = plot_roc_curve(classifier, X[test], y[test],
                                     label=None,
                                     alpha=0.3, lw=1, ax=ax_roc, c="tab:blue")
                precision, recall, _ = precision_recall_curve(y_binary[test], classifier.predict_proba(X[test])[:, 1])
                ax_pr.step(recall, precision, label=None, lw=1, c="tab:blue")

            interp_tpr = np.interp(mean_fpr, viz_roc.fpr, viz_roc.tpr)
            interp_tpr[0] = 0.0
            print("auc=", viz_roc.roc_auc)
            # if "PCA(2)" in steps:
            #     plot_2D_decision_boundaries(viz_roc.roc_auc, i, X, y, X[test], y[test], X[train], y[train], steps, classifier, out_dir, steps)
            if "PCA(3)" in steps:
                plot_3D_decision_boundaries(X, y, X[train], y[train], X[test], y[test], steps, classifier, i, out_dir, steps, viz_roc.roc_auc)
            if np.isnan(viz_roc.roc_auc):
                continue
            tprs.append(interp_tpr)
            aucs_roc.append(viz_roc.roc_auc)
            aucs_pr.append(auc(recall, precision))
            precisions.append(precision)
            recalls.append(recall)

            y_ground_truth_pr.append(y_binary[test])
            y_proba_pr.append(classifier.predict_proba(X[test])[:, 1])

            # ax.plot(viz.fpr, viz.tpr, c="tab:green")
        print("make_roc_curve done!")
        mean_auc = plot_roc_range(ax_roc, tprs, mean_fpr, aucs_roc, out_dir, steps, fig_roc, cv_name, days)
        mean_auc_pr = plot_pr_range(ax_pr, y_ground_truth_pr, y_proba_pr, aucs_pr, out_dir, steps, fig_pr, cv_name, days)

        plt.close(fig_roc)
        plt.close(fig_pr)
        plt.clf()
        return mean_auc, aucs_roc


def downsampleDf(data_frame, class_healthy, class_unhealthy):
    df_true = data_frame[data_frame['target'] == class_unhealthy]
    df_false = data_frame[data_frame['target'] == class_healthy]
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


def setupGraphOutputPath(output_dir):
    graph_outputdir = "%s/input_graphs/" % output_dir
    # if os.path.exists(graph_outputdir):
    #     print("purge %s ..." % graph_outputdir)
    #     try:
    #         shutil.rmtree(graph_outputdir)
    #     except IOError:
    #         print("file not found.")
    create_rec_dir(graph_outputdir)
    return graph_outputdir


def applyPreprocessingSteps(df_hum, df_temp, sfft_window, wavelet_f0, animal_ids, df, N_META, output_dir, steps, class_healthy_label, class_unhealthy_label,
                            class_healthy, class_unhealthy, clf_name="", output_dim=2, n_scales=None):
    step_slug = "_".join(steps)
    graph_outputdir = setupGraphOutputPath(output_dir) + "/" + clf_name + "/" + step_slug

    if len(steps) == 0:
        print("no steps to apply! return data as is")
        return df
    print("BEFORE STEP ->", df)
    # plotDistribution(df.iloc[:, :-N_META].values, graph_outputdir, "data_distribution_before_%s" % step_slug)
    for step in steps:
        if step not in ["ANSCOMBE", "LOG", "QN", "CWT", "CENTER", "MINMAX", "PCA", "BASELINERM", "STFT", "STANDARDSCALER"]:
            warnings.warn("processing step %s does not exist!" % step)
        #plotDistribution(df.iloc[:, :-N_META].values, graph_outputdir, "data_distribution_before_%s" % step)
        print("applying STEP->%s in [%s]..." % (step, step_slug.replace("_", "->")))
        if step == "TEMPERATUREAPPEND":
            df_activity = df.copy().iloc[:, :-N_META]
            df_activity.index = df.index  # need to keep original sample index!!!!
            df_meta = df.iloc[:, -N_META:]
            df_temp = df_temp.loc[df.index]
            df = pd.concat([df_activity, df_temp, df_meta], axis=1)
            new_header = [str(x) for x in np.arange(df.shape[1]-N_META)] + df.columns[df.shape[1]-N_META:].tolist()
            df.columns = new_header

        if step == "HUMIDITYAPPEND":
            df_activity = df.copy().iloc[:, :-N_META]
            df_activity.index = df.index  # need to keep original sample index!!!!
            df_meta = df.iloc[:, -N_META:]
            df_hum = df_hum.loc[df.index]
            df = pd.concat([df_activity, df_hum, df_meta], axis=1)
            new_header = [str(x) for x in np.arange(df.shape[1]-N_META)] + df.columns[df.shape[1]-N_META:].tolist()
            df.columns = new_header

        if step == "TEMPERATURE":
            df_activity = df.copy().iloc[:, :-N_META]
            df_activity.index = df.index  # need to keep original sample index!!!!
            df_meta = df.iloc[:, -N_META:]
            df_temp = df_temp.loc[df.index]
            df = pd.concat([df_temp, df_meta], axis=1)
            new_header = [str(x) for x in np.arange(df.shape[1]-N_META)] + df.columns[df.shape[1]-N_META:].tolist()
            df.columns = new_header

        if step == "HUMIDITY":
            df_activity = df.copy().iloc[:, :-N_META]
            df_activity.index = df.index  # need to keep original sample index!!!!
            df_meta = df.iloc[:, -N_META:]
            df_hum = df_hum.loc[df.index]
            df = pd.concat([df_hum, df_meta], axis=1)
            new_header = [str(x) for x in np.arange(df.shape[1]-N_META)] + df.columns[df.shape[1]-N_META:].tolist()
            df.columns = new_header

        if step == "BASELINERM":
            df.iloc[:, :-N_META] = BaseLineScaler().fit_transform(df.iloc[:, :-N_META].values)
        if step == "STANDARDSCALER":
            df.iloc[:, :-N_META] = StandardScaler(with_mean=False, with_std=True).fit_transform(
                df.iloc[:, :-N_META].values)

            # if "TEMPERATURE" not in step_slug and "HUMIDITY" not in step_slug and "PCA" not in step_slug:
            #     if "CWT" in step_slug:
            #         SampleVisualisation(df, CWT_Transform.shape, N_META, graph_outputdir + "/" + step, step_slug, None, None, CWT_Transform.scales)
            #
            #     if "STFT" in step_slug and "PCA" not in step_slug:
            #         SampleVisualisation(df, STFT_Transform.shape, N_META, graph_outputdir + "/" + step, step_slug,
            #                             STFT_Transform.sfft_window, STFT_Transform.stft_time, STFT_Transform.freqs)

        if step == "CENTER":
            df.iloc[:, :-N_META] = CenterScaler(center_by_sample=False).fit_transform(df.iloc[:, :-N_META].values)
        if step == "CENTER_STD":
            df.iloc[:, :-N_META] = CenterScaler(center_by_sample=True, divide_by_std=True).fit_transform(df.iloc[:, :-N_META].values)
        if step == "MINMAX":
            df.iloc[:, :-N_META] = MinMaxScaler().fit_transform(df.iloc[:, :-N_META].values)
        if step == "ANSCOMBE":
            df.iloc[:, :-N_META] = Anscombe().transform(df.iloc[:, :-N_META].values)
        if step == "SQRT":
            df.iloc[:, :-N_META] = Sqrt().transform(df.iloc[:, :-N_META].values)
        if step == "LOG":
            df.iloc[:, :-N_META] = Log().transform(df.iloc[:, :-N_META].values)
        if step == "QN":
            df.iloc[:, :-N_META] = QuotientNormalizer(out_dir=graph_outputdir + "/" +step).transform(df.iloc[:, :-N_META].values)
        if "STFT" in step:
            STFT_Transform = STFT(sfft_window=sfft_window, out_dir=graph_outputdir + "/" + step, step_slug=step_slug,
                                  animal_ids=animal_ids, targets=df["target"].tolist(),
                                dates=df["date"].tolist())
            d = STFT_Transform.transform(df.copy().iloc[:, :-N_META].values)
            data_frame_stft = pd.DataFrame(d)
            data_frame_stft.index = df.index  # need to keep original sample index!!!!
            df_meta = df.iloc[:, -N_META:]
            df = pd.concat([data_frame_stft, df_meta], axis=1)
            del data_frame_stft
        if "CWT" in step:
            df_meta = df.iloc[:, -N_META:]
            df_o = df.copy()
            CWT_Transform = CWT(wavelet_f0=wavelet_f0, out_dir=graph_outputdir + "/" + step, step_slug=step_slug,
                                n_scales=n_scales, animal_ids=animal_ids, targets=df["target"].tolist(),
                                dates=df["date"].tolist())
            data_frame_cwt, data_frame_cwt_raw = CWT_Transform.transform(df.copy().iloc[:, :-N_META].values)
            data_frame_cwt = pd.DataFrame(data_frame_cwt)
            data_frame_cwt_raw = pd.DataFrame(data_frame_cwt_raw)

            # data_frame_cwt.index = df.index  # need to keep original sample index!!!!
            # df_meta = df.iloc[:, -N_META:]
            # df = pd.concat([data_frame_cwt, df_meta], axis=1)
            # sanity check#################################################################################################
            #wont work sincce using avg of sample!
            # rdm_idxs = random.choices(df.index.tolist(), k=1)
            # samples_tocheck = df_o.loc[(rdm_idxs), :].values[:, :-N_META]
            # cwt_to_check = pd.DataFrame(CWT(out_dir=graph_outputdir + "/" + step + "/cwt_sanity_check/").transform(samples_tocheck))
            # prev_cwt_results = df.loc[(rdm_idxs), :].values[:, :-N_META]
            # assert False not in (cwt_to_check.values == prev_cwt_results), "missmatch in cwt sample!"
            #############################################################################################################

            data_frame_cwt.index = df.index# need to keep original sample index!!!!
            CWTVisualisation(step_slug, graph_outputdir, CWT_Transform.shape, CWT_Transform.coi_mask, CWT_Transform.scales, CWT_Transform.coi, df_o.copy(),
                             data_frame_cwt, class_healthy_label, class_unhealthy_label, class_healthy, class_unhealthy)

            data_frame_cwt_raw.index = df.index  # need to keep original sample index!!!!
            df = pd.concat([data_frame_cwt_raw, df_meta], axis=1)
            # CWTVisualisation(step_slug, graph_outputdir, CWT_Transform.shape, CWT_Transform.coi_mask, CWT_Transform.scales, CWT_Transform.coi, df_o.copy(),
            #                  data_frame_cwt_raw, class_healthy_label, class_unhealthy_label, class_healthy, class_unhealthy, filename_sub="real")

            df = df.dropna(axis=1, how='all') #removes nan from coi
            del data_frame_cwt
            del data_frame_cwt_raw
        if "PCA" in step:
            pca_dim = int(step[step.find("(")+1:step.find(")")])
            print("pca_dim", pca_dim)
            df_before_reduction = df.iloc[:, :-N_META].values
            data_frame_pca = pd.DataFrame(PCA(n_components=pca_dim).fit_transform(df_before_reduction))
            data_frame_pca.index = df.index  # need to keep original sample index!!!!
            df_meta = df.iloc[:, -N_META:]
            df = pd.concat([data_frame_pca, df_meta], axis=1)
            del data_frame_pca

        print("AFTER STEP ->", df)
        if "CWT" not in step_slug:
            plotDistribution(df.iloc[:, :-N_META].values, graph_outputdir, "data_distribution_after_%s" % step)

    # if "PCA" in step_slug:
    #     plotDistribution(df.iloc[:, :-N_META].values, graph_outputdir, "data_distribution_after_%s" % step_slug)
    return df


def loadActivityData(filepath, day):
    print("load activity from datasets...", filepath)
    data_frame = pd.read_csv(filepath, sep=",", header=None, low_memory=False)
    data_frame = data_frame.astype(dtype=float, errors='ignore')  # cast numeric values as float
    data_point_count = data_frame.shape[1]
    hearder = [str(n) for n in range(0, data_point_count)]
    N_META = 4
    hearder[-4] = 'label'
    hearder[-3] = 'id'
    hearder[-2] = 'imputed_days'
    hearder[-1] = 'date'
    data_frame.columns = hearder
    data_frame = data_frame[~np.isnan(data_frame["imputed_days"])]
    data_frame = data_frame.fillna(-1)
    # filter with imputed_days count
    data_frame = data_frame[data_frame["imputed_days"] >= day]

    #1To1 1To2 2To2 2To1
    new_label = []
    for v in data_frame["label"].values:
        if v in ["1To1"]:
            new_label.append("1To1")
            continue
        if v in ["2To4", "3To4", "1To4", "1To3", "4To5", "2To3"]:
            new_label.append("1To2")
            continue
        new_label.append(v)

    data_frame["label"] = new_label

    return data_frame, N_META


def process_data_frame_1dcnn(epochs, stratify, animal_ids, output_dir, data_frame, days, farm_id, steps, n_splits, n_repeats, sampling,
                       downsample_false_class, label_series, class_healthy, class_unhealthy, y_col='target',
                       cv="StratifiedLeaveTwoOut"):
    print(label_series)
    data_frame["id"] = animal_ids
    data_frame = data_frame.loc[data_frame['target'].isin([class_healthy, class_unhealthy])]
    if downsample_false_class:
        data_frame = downsampleDf(data_frame, class_healthy, class_unhealthy)

    sample_idxs = data_frame.index.tolist()

    if cv == "StratifiedLeaveTwoOut":
        cross_validation_method = StratifiedLeaveTwoOut(animal_ids, sample_idxs, stratified=True, verbose=True)

    if cv == "LeaveTwoOut":
        cross_validation_method = StratifiedLeaveTwoOut(animal_ids, sample_idxs, stratified=False, verbose=True)

    if cv == "RepeatedStratifiedKFold":
        cross_validation_method = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)

    if cv == "RepeatedKFold":
        cross_validation_method = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)

    if cv == "LeaveOneOut":
        cross_validation_method = LeaveOneOut()

    data_frame = data_frame.drop("id", 1)

    y = data_frame[y_col].values.flatten()
    y = y.astype(int)
    X = data_frame[data_frame.columns[0:data_frame.shape[1] - 1]].values
    run1DCnn(epochs, cross_validation_method, X, y, class_healthy, class_unhealthy, steps,
             days, farm_id, sampling, label_series, downsample_false_class, output_dir, cv)


def process_data_frame_2dcnn(wavelet_f0, epochs, stratify, animal_ids, output_dir, data_frame, days, farm_id, steps, n_splits, n_repeats, sampling,
                       downsample_false_class, label_series, class_healthy, class_unhealthy, y_col='target',
                       cv="StratifiedLeaveTwoOut"):
    print(label_series)
    data_frame["id"] = animal_ids
    data_frame = data_frame.loc[data_frame['target'].isin([class_healthy, class_unhealthy])]
    if downsample_false_class:
        data_frame = downsampleDf(data_frame, class_healthy, class_unhealthy)

    sample_idxs = data_frame.index.tolist()

    if cv == "StratifiedLeaveTwoOut":
        cross_validation_method = StratifiedLeaveTwoOut(animal_ids, sample_idxs, stratified=True, verbose=True)

    if cv == "LeaveTwoOut":
        cross_validation_method = StratifiedLeaveTwoOut(animal_ids, sample_idxs, stratified=False, verbose=True)

    if cv == "RepeatedStratifiedKFold":
        cross_validation_method = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)

    if cv == "RepeatedKFold":
        cross_validation_method = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)

    if cv == "LeaveOneOut":
        cross_validation_method = LeaveOneOut()

    data_frame = data_frame.drop("id", 1)

    y = data_frame[y_col].values.flatten()
    y = y.astype(int)
    X = data_frame[data_frame.columns[0:data_frame.shape[1] - 1]].values
    run2DCnn(wavelet_f0, epochs, cross_validation_method, X, y, class_healthy, class_unhealthy, steps,
             days, farm_id, sampling, label_series, downsample_false_class, output_dir)


def process_data_frame_svm(output_dir, stratify, animal_ids, out_dir, data_frame, days, farm_id, steps, n_splits, n_repeats, sampling,
                           downsample_false_class, label_series, class_healthy, class_unhealthy, y_col='target',
                           cv=None):
    print("*******************************************************************")
    mlp_layers = (1000, 500, 100, 45, 30, 15)
    print(label_series)
    data_frame["id"] = animal_ids
    data_frame = data_frame.loc[data_frame['target'].isin([class_healthy, class_unhealthy])]
    if downsample_false_class:
        data_frame = downsampleDf(data_frame, class_healthy, class_unhealthy)

    #animal_ids = data_frame["id"].tolist()
    sample_idxs = data_frame.index.tolist()

    if cv == "StratifiedLeaveTwoOut":
        cross_validation_method = StratifiedLeaveTwoOut(animal_ids, sample_idxs, stratified=True, verbose=True)

    if cv == "LeaveTwoOut":
        cross_validation_method = StratifiedLeaveTwoOut(animal_ids, sample_idxs, stratified=False, verbose=True)

    if cv == "RepeatedStratifiedKFold":
        cross_validation_method = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=None)

    if cv == "RepeatedKFold":
        cross_validation_method = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)

    if cv == "LeaveOneOut":
        cross_validation_method = LeaveOneOut()

    data_frame = data_frame.drop("id", 1)

    y = data_frame[y_col].values.flatten()
    y = y.astype(int)
    X = data_frame[data_frame.columns[0:data_frame.shape[1] - 1]].values


    print("release data_frame memory...")
    del data_frame
    gc.collect()
    print("****************************")

    if not os.path.exists(output_dir):
        print("mkdir", output_dir)
        os.makedirs(output_dir)

    plotHeatmap(X, output_dir, "CLF_INPUT_%s" % steps, "CLF_INPUT_%s.html" % steps, xaxis="features", yaxis="value")

    filename_2d_scatter = "%s/PLS/%s_2DPLS_days_%d_option_%s_downsampled_%s_sampling_%s.png" % (
        output_dir, farm_id, days, steps, downsample_false_class, sampling)

    pls = PLSRegression(n_components=2)
    X_pls = pls.fit_transform(X.copy(), y.copy())[0]
    plot_2d_space(X_pls, y, filename_2d_scatter, label_series, '2 PLS components ' + steps)

    print("************************************************")
    print("downsample on= " + str(downsample_false_class))
    class0_count = str(y[y == class_healthy].size)
    class1_count = str(y[y == class_unhealthy].size)
    print("X-> class0=" + class0_count + " class1=" + class1_count)
    try:
        if int(class1_count) < 2 or int(class0_count) < 2:
            print("not enough samples!")
            return
    except ValueError as e:
        print(e)
        return

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



    # param_str = "option_%s_downsample_%s_days_%d_farmid_%s_nrepeat_%d_nsplits_%d_class0_%s_class1_%s_sampling_%s" % (
    #     steps, str(downsample_false_class), days, farm_id, n_repeats, n_splits, class0_count,
    #     class1_count, sampling)
    report_rows_list = []

    for clf_svc in [SVC(kernel="linear", probability=True, class_weight='balanced')]:
        # tuned_parameters = [{'kernel': ['rbf'], 'gamma': ['scale', 1e-1, 1e-3, 1e-4], 'class_weight': [None, 'balanced'],
        #                      'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]},
        #                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
        # clf = GridSearchCV(clf_svc, tuned_parameters, cv=cross_validation_method, scoring='roc_auc', n_jobs=-1)
        # clf.fit(X.copy(), y.copy())
        # clf_best = clf.best_estimator_
        # print("Best estimator from gridsearch=")
        # print(clf_best)
        scores = cross_validate(clf_svc, X.copy(), y.copy(), cv=cross_validation_method, scoring=scoring, n_jobs=-1)
        scores["downsample"] = downsample_false_class
        scores["class0"] = y[y == class_healthy].size
        scores["class1"] = y[y == class_unhealthy].size
        scores["steps"] = steps
        scores["days"] = days
        scores["farm_id"] = farm_id
        scores["balanced_accuracy_score_mean"] = np.mean(scores["test_balanced_accuracy_score"])
        scores["precision_score0_mean"] = np.mean(scores["test_precision_score0"])
        scores["precision_score1_mean"] = np.mean(scores["test_precision_score1"])
        scores["recall_score0_mean"] = np.mean(scores["test_recall_score0"])
        scores["recall_score1_mean"] = np.mean(scores["test_recall_score1"])
        scores["f1_score0_mean"] = np.mean(scores["test_f1_score0"])
        scores["f1_score1_mean"] = np.mean(scores["test_f1_score1"])
        scores["sampling"] = sampling
        scores["classifier"] = "->SVC(%s)" % clf_svc.kernel
        scores["classifier_details"] = str(clf_svc).replace('\n', '').replace(" ", '')
        #clf_svc = make_pipeline(SVC(probability=True, class_weight='balanced'))
        auc_m, aucs = makeRocCurve(scores["classifier"].replace("->", ""), out_dir, clf_svc, X.copy(), y.copy(), cross_validation_method, steps, cv, animal_ids, days)
        scores["roc_auc_score_mean"] = auc_m
        scores["roc_auc_scores"] = aucs
        report_rows_list.append(scores)

        df_report = pd.DataFrame(report_rows_list)
        df_report["class_0_label"] = label_series[class_healthy]
        df_report["class_1_label"] = label_series[class_unhealthy]
        df_report["nfold"] = cross_validation_method.nfold if hasattr(cross_validation_method, 'nfold') else np.nan
        # df_report["n_splits"] = cross_validation_method.cvargs['n_splits'] if hasattr(cross_validation_method,
        #                                                                               'cvargs') else np.nan
        # df_report["n_repeats"] = cross_validation_method.n_repeats if hasattr(cross_validation_method,
        #                                                                       'n_repeats') else np.nan
        df_report["total_fit_time"] = [time.strftime('%H:%M:%S', time.gmtime(np.nansum(x))) for x in
                                       df_report["fit_time"].values]
        filename = "%s/%s/%s_%s_classification_report_days_%d_option_%s_downsampled_%s_sampling_%s.csv" % (
            output_dir, cv, scores["classifier"].replace("->", ""), farm_id, days, steps, downsample_false_class, sampling)
        create_rec_dir(filename)
        df_report.to_csv(filename, sep=',', index=False)
        print("filename=", filename)
        del scores


def parse_param_from_filename(file):
    split = file.split("/")[-1].split('.')[0].split('_')
    # activity_delmas_70101200027_dbft_1_1min
    sampling = split[5]
    days = int(split[4])
    farm_id = split[1] + "_" + split[2]
    option = split[0]
    return days, farm_id, option, sampling


def main(preprocessing_steps, output_dir, dataset_folder, class_healthy, class_unhealthy, stratify, n_scales,
         hum_file, temp_file, n_splits, n_repeats, epochs, n_process, output_samples, output_cwt, cv, wavelet_f0, sfft_window):
    print("output_dir=", output_dir)
    print("dataset_filepath=", dataset_folder)
    print("class_healthy=", class_healthy)
    print("class_unhealthy=", class_unhealthy)
    print("output_samples=", output_samples)
    print("stratify=", stratify)
    print("output_cwt=", output_cwt)
    print("hum_file=", hum_file)
    print("temp_file=", temp_file)
    print("epochs=", epochs)
    print("n_process=", n_process)
    print("output_samples=", output_samples)
    print("output_cwt=", output_cwt)
    print("cv=", cv)
    print("wavelet_f0=", wavelet_f0)
    print("sfft_window=", sfft_window)
    print("loading dataset...")
    enable_downsample_df = False
    day = int(dataset_folder.split('_')[-1][0])

    files = glob.glob(dataset_folder + "/*.csv")  # find datset files
    files = [file.replace("\\", '/') for file in files]
    print("found %d files." % len(files))
    print(files)

    has_humidity_data = False
    df_hum = None
    if hum_file is not None:
        has_humidity_data = True
        print("humidity file detected!", hum_file)
        df_hum = pd.read_csv(hum_file)
        print(df_hum.shape)
        plotHeatmap(df_hum.values, output_dir, "Samples humidity", "humidity.html")

    has_temperature_data = True
    df_temp = None
    if temp_file is not None:
        has_temperature_data = True
        print("temperature file detected!", temp_file)
        df_temp = pd.read_csv(temp_file)
        plotHeatmap(df_temp.values, output_dir, "Samples temperature", "temperature.html")
        print(df_temp.shape)

    has_humidity_and_temp = False
    df_hum_temp = None
    if temp_file is not None and hum_file is not None:
        has_humidity_and_temp = True
        print("temperature file detected!", temp_file)
        print("humidity file detected!", hum_file)
        df_hum_temp = pd.concat([df_temp, df_hum], axis=1)
        plotHeatmap(df_hum_temp.values, output_dir, "Samples temperature and Humidity", "temperature_humidity.html")
        print(df_hum_temp.shape)

    for file in files:
        days, farm_id, option, sampling = parse_param_from_filename(file)
        print("loading dataset file %s ..." % file)
        data_frame, N_META = loadActivityData(file, day)

        data_frame_o = data_frame.copy()
        print(data_frame)

        # Hot Encode of FAmacha targets and assign integer target to each famacha label
        data_frame_labeled = pd.get_dummies(data_frame, columns=["label"])
        flabels = [x for x in data_frame_labeled.columns if 'label' in x]
        data_frame["target"] = 0
        for i, flabel in enumerate(flabels):
            data_frame_labeled[flabel] = data_frame_labeled[flabel] * (i + 1)
            data_frame["target"] = data_frame["target"] + data_frame_labeled[flabel]
        class_count = {}
        label_series = dict(data_frame[['target', 'label']].drop_duplicates().values)
        label_series_inverse = dict((v, k) for k, v in label_series.items())
        class_healthy = label_series_inverse[class_healthy]
        class_unhealthy = label_series_inverse[class_unhealthy]
        print(label_series)
        class_healthy_label = label_series[class_healthy]
        class_unhealthy_label = label_series[class_unhealthy]
        for k in label_series.keys():
            class_count[label_series[k] + "_" + str(k)] = data_frame[data_frame['target'] == k].shape[0]
        print(class_count)
        # drop label column stored previously, just keep target for ml
        data_frame = data_frame.drop('label', 1)
        print(data_frame)

        #plotMeanGroups(n_scales, wavelet_f0, data_frame, label_series, N_META, output_dir + "/raw_before_qn/")
        ################################################################################################################
        ##VISUALISATION
        ################################################################################################################
        animal_ids = data_frame.iloc[0:len(data_frame), :]["id"].astype(str).tolist()
        df_norm = applyPreprocessingSteps(df_hum, df_temp, sfft_window, wavelet_f0, animal_ids, data_frame.copy(), N_META, output_dir, ["QN"],
                                          class_healthy_label, class_unhealthy_label, class_healthy, class_unhealthy,
                                          clf_name="SVM_QN_VISU", n_scales=n_scales)
        plot_zeros_distrib(label_series, df_norm, output_dir,
                           title='Percentage of zeros in activity per sample after normalisation')
        plot_zeros_distrib(label_series, data_frame.copy(), output_dir,
                           title='Percentage of zeros in activity per sample before normalisation')
        plotMeanGroups(n_scales, sfft_window, wavelet_f0, df_norm, label_series, N_META, output_dir + "/raw_after_qn/")

        plot_time_pca(N_META, data_frame.copy(), output_dir, label_series, title="PCA time domain before normalisation")
        plot_time_pca(N_META, df_norm, output_dir, label_series, title="PCA time domain after normalisation")

        plot_time_lda(N_META, data_frame.copy(), output_dir, label_series, title="LDA time domain before normalisation")
        plot_time_lda(N_META, data_frame.copy(), output_dir, label_series, title="LDA time domain after normalisation")

        ntraces = 2
        idx_healthy, idx_unhealthy = plot_groups(N_META, animal_ids, class_healthy_label, class_unhealthy_label,
                                                 class_healthy,
                                                 class_unhealthy, output_dir, data_frame.copy(), title="Raw imputed",
                                                 xlabel="Time",
                                                 ylabel="activity", ntraces=ntraces)
        plot_groups(N_META, animal_ids, class_healthy_label, class_unhealthy_label, class_healthy, class_unhealthy,
                    output_dir,
                    df_norm, title="Normalised(Quotient Norm) samples", xlabel="Time", ylabel="activity",
                    idx_healthy=idx_healthy, idx_unhealthy=idx_unhealthy, stepid=2, ntraces=ntraces)
        ################################################################################################################
        # keep only two class of samples
        data_frame = data_frame[data_frame["target"].isin([class_healthy, class_unhealthy])]
        animal_ids = data_frame.iloc[0:len(data_frame), :]["id"].astype(str).tolist()
        # cv = "StratifiedLeaveTwoOut"

        for steps in preprocessing_steps:
            step_slug = "_".join(steps)
            df_processed = applyPreprocessingSteps(df_hum, df_temp, sfft_window, wavelet_f0, animal_ids, data_frame.copy(), N_META, output_dir, steps,
                                                   class_healthy_label, class_unhealthy_label, class_healthy,
                                                   class_unhealthy, clf_name="SVM", output_dim=data_frame.shape[0],
                                                   n_scales=n_scales)
            targets = df_processed["target"]
            df_processed = df_processed.iloc[:, :-N_META]
            df_processed["target"] = targets
            process_data_frame_svm(output_dir, stratify, animal_ids, output_dir, df_processed, days, farm_id, step_slug,
                                   n_splits, n_repeats,
                                   sampling, enable_downsample_df, label_series, class_healthy, class_unhealthy,
                                   cv=cv)


        #2DCNN
        # for steps in [["QN", "ANSCOMBE", "LOG"]]:
        #     step_slug = "_".join(steps)
        #     step_slug = step_slug + "_2DCNN"
        #     df_processed = applyPreprocessingSteps(data_frame.copy(), N_META, output_dir, steps,
        #                                            class_healthy_label, class_unhealthy_label, class_healthy, class_unhealthy, clf_name="2DCNN")
        #     targets = df_processed["target"]
        #     df_processed = df_processed.iloc[:, :-N_META]
        #     df_processed["target"] = targets
        #     process_data_frame_2dcnn(epochs, stratify, animal_ids, output_dir, df_processed, days, farm_id, step_slug, n_splits, n_repeats, sampling,
        #                    enable_downsample_df, label_series, class_healthy, class_unhealthy, cv="StratifiedLeaveTwoOut")

        #1DCNN
        # for steps in [["QN"], ["QN", "ANSCOMBE", "LOG"]]:
        #     step_slug = "_".join(steps)
        #     step_slug = step_slug + "_1DCNN"
        #     df_processed = applyPreprocessingSteps(sfft_window, wavelet_f0, animal_ids, data_frame.copy(), N_META, output_dir, steps,
        #                                            class_healthy_label, class_unhealthy_label, class_healthy, class_unhealthy, clf_name="1DCNN", output_dim=data_frame.shape[0])
        #     targets = df_processed["target"]
        #     df_processed = df_processed.iloc[:, :-N_META]
        #     df_processed["target"] = targets
        #     process_data_frame_1dcnn(epochs, stratify, animal_ids, output_dir, df_processed, days, farm_id, step_slug, n_splits, n_repeats, sampling,
        #                    enable_downsample_df, label_series, class_healthy, class_unhealthy, cv=cv)



    output_dir = "%s/%s" % (output_dir, cv)
    files = [output_dir + "/" + file for file in os.listdir(output_dir) if file.endswith(".csv")]
    print("found %d files." % len(files))
    print("compiling final file...")
    df_final = pd.DataFrame()
    dfs = [pd.read_csv(file, sep=",") for file in files]
    df_final = pd.concat(dfs)
    filename = "%s/final_classification_report.csv" % output_dir
    df_final.to_csv(filename, sep=',', index=False)
    print(df_final)
    plotMlReport(filename, output_dir)


if __name__ == "__main__":
    print("********************************************************************")
    print("*                          ML PIPELINE                             *")
    print("********************************************************************")
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', help='output directory', type=str)
    parser.add_argument('dataset_folder', help='dataset input directory', type=str)
    parser.add_argument('--class_healthy', help='label for healthy class', default="1To1", type=str)
    parser.add_argument('--class_unhealthy', help='label for unhealthy class', default="1To2", type=str)
    parser.add_argument('--stratify', help='enable stratiy for cross validation', default='n', type=str)
    parser.add_argument('--s_output', help='output sample files', default='y', type=str)
    parser.add_argument('--cwt', help='enable freq domain (cwt)', default='y', type=str)
    parser.add_argument('--n_scales', help='n scales in dyadic array [2^2....2^n].', default=10, type=int)
    parser.add_argument('--temp_file', help='temperature features.', default=None, type=str)
    parser.add_argument('--hum_file', help='humidity features.', default=None, type=str)
    parser.add_argument('--n_splits', help='number of splits for repeatedkfold cv', default=5, type=int)
    parser.add_argument('--n_repeats', help='number of repeats for repeatedkfold cv', default=10, type=int)
    parser.add_argument('--cv', help='cross validation method (LeaveTwoOut|StratifiedLeaveTwoOut|RepeatedStratifiedKFold| RepeatedKFold|LeaveOneOut)',
                        default="RepeatedKFold", type=str)
    parser.add_argument('--wavelet_f0', help='Mother Wavelet frequency for CWT', default=6, type=int)
    parser.add_argument('--sfft_window', help='STFT window size', default=60, type=int)
    parser.add_argument('--epochs', help='cnn epochs', default=20, type=int)
    parser.add_argument('--n_process', help='number of threads to use.', default=6, type=int)

    args = parser.parse_args()

    output_dir = args.output_dir
    dataset_folder = args.dataset_folder
    class_healthy = args.class_healthy
    class_unhealthy = args.class_unhealthy
    stratify = args.stratify
    s_output = args.s_output
    cwt = args.cwt
    n_scales = args.n_scales
    hum_file = args.hum_file
    temp_file = args.temp_file
    n_splits = args.n_splits
    n_repeats = args.n_repeats
    epochs = args.epochs
    n_process = args.n_process
    cv = args.cv
    wavelet_f0 = args.wavelet_f0
    sfft_window = args.sfft_window

    stratify = "y" in stratify.lower()
    output_samples = "y" in s_output.lower()
    output_cwt = "y" in cwt.lower()
    #
    # steps = [
    #          ["QN", "ANSCOMBE", "LOG"],
    #          ["QN", "ANSCOMBE", "LOG", "CENTER", "STFT", "STANDARDSCALER"],
    #          ["QN", "ANSCOMBE", "LOG", "CENTER", "CWT(MEXH)", "STANDARDSCALER"],
    #          ["QN", "ANSCOMBE", "LOG", "CENTER", "CWT(MORL)", "STANDARDSCALER"],
    #          ["QN", "ANSCOMBE", "LOG", "CENTER", "CWT(MEXH)", "STANDARDSCALER", "PCA(2)"],
    #          ["QN", "ANSCOMBE", "LOG", "CENTER", "CWT(MORL)", "STANDARDSCALER", "PCA(2)"],
    #          ]

    steps = [
             ["QN", "ANSCOMBE", "LOG", "PCA(2)"],
             ["QN", "ANSCOMBE", "LOG", "PCA(3)"],
             ["QN", "ANSCOMBE", "LOG", "CENTER", "CWT(MORL)", "STANDARDSCALER", "PCA(2)"],
             ["QN", "ANSCOMBE", "LOG", "CENTER", "CWT(MORL)", "STANDARDSCALER", "PCA(3)"],
             ]


    main(steps, output_dir, dataset_folder, class_healthy, class_unhealthy, stratify, n_scales,
         hum_file, temp_file, n_splits, n_repeats, epochs, n_process, output_samples, output_cwt, cv, wavelet_f0, sfft_window)