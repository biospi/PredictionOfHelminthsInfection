from sklearn import datasets, linear_model
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.metrics import recall_score, balanced_accuracy_score, roc_auc_score, precision_score, f1_score, roc_curve
import numpy as np
from sys import exit
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import auc
from sklearn.datasets import make_blobs


def mean_confidence_interval(x):
    # boot_median = [np.median(np.random.choice(x, len(x))) for _ in range(iteration)]
    x.sort()
    lo_x_boot = np.percentile(x, 2.5)
    hi_x_boot = np.percentile(x, 97.5)
    # print(lo_x_boot, hi_x_boot)
    return lo_x_boot, hi_x_boot


def plot_roc_range(ax, tprs, mean_fpr, aucs, out_dir, classifier_name, fig):
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='orange',
            label='Chance', alpha=1)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    # std_auc = np.std(aucs)
    lo, hi = mean_confidence_interval(aucs)
    label = r'Mean ROC (Mean AUC = %0.2f, 95%% CI [%0.4f, %0.4f] )' % (mean_auc, lo, hi)
    if len(aucs) <= 2:
        label = r'Mean ROC (Mean AUC = %0.2f)' % mean_auc
    ax.plot(mean_fpr, mean_tpr, color='tab:blue',
            label=label,
            lw=2, alpha=.8)

    # std_tpr = np.std(tprs, axis=0)
    # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    #
    # confidence_lower, confidence_upper = get_conf_interval(tprs, mean_fpr)

    # ax.fill_between(mean_fpr, confidence_lower, confidence_upper, color='tab:blue', alpha=.2)
    #                 #label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="Receiver operating characteristic iteration")
    ax.legend(loc="lower right")
    fig.show()
    return mean_auc

def make_roc_curve(out_dir, classifier, X, y, cv, param_str):
    print("make_roc_curve")

    if isinstance(X, pd.DataFrame):
        X = X.values
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    plt.clf()
    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X[train], y[train])
        viz = plot_roc_curve(classifier, X[test], y[test],
                             label=None,
                             alpha=0.3, lw=1, ax=ax, c="tab:blue")
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        # ax.plot(viz.fpr, viz.tpr, c="tab:green")
    # clf_name = "%s_%s" % ("_".join([x[0] for x in classifier.steps]), param_str)

    mean_auc = plot_roc_range(ax, tprs, mean_fpr, aucs, out_dir, "clf_name", fig)

    plt.close(fig)
    plt.clf()
    aucs2 = np.mean(aucs)
    return mean_auc, aucs2

from plotnine import *
import glob
from pathlib import Path
import json
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


if __name__ == "__main__":

    DIR = "F:/Data2/imp_full_reshape_andytcoldate_debug12"

    rmse_list = []
    rmse_list_li = []
    files = []
    for path in Path(DIR).rglob('*.json'):
        files.append(str(path))

    files.sort(key=natural_keys)

    for path in files:
        print(path)
        with open(path) as json_file:
            data = json.load(json_file)
            print(data["rmse"])
            rmse_list.append(data["rmse"])
            rmse_list_li.append(data["rmse_li"])

    plt.clf()
    plt.cla()
    fig, ax = plt.subplots()
    ax.set_ylabel('RMSE')
    ax.set_xlabel('iteration')
    plt.plot([x * 10 for x in range(len(rmse_list))], rmse_list, label="RMSE GAIN", alpha=1, marker='*')
    plt.plot([x * 10 for x in range(len(rmse_list_li))], rmse_list_li, label="RMSE LI", alpha=1, marker='*')

    plt.title("RMSE iteration performance")
    plt.legend()
    plt.show()


    exit()


    df = pd.read_csv("z_prct_data.csv")
    g = (ggplot(df)  # defining what data to use
     + aes(x='Target', y='Percent of zeros', color='Target', shape='Target')  # defning what variable to usei
     + geom_jitter()  # defining the type of plot to use
     + stat_summary(geom="crossbar", color="black", width=0.2)
     + theme(subplots_adjust={'right': 0.82})
     )

    fig = g.draw()
    fig.tight_layout()
    fig.show()

    exit(0)



    # generate 2d classification dataset
    X, y = make_blobs(n_samples=100, centers=2, n_features=2)
    # scatter plot, dots colored by class value
    df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
    colors = {0: 'red', 1: 'blue', 2: 'green'}
    fig, ax = plt.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    plt.show()

    scoring = {
        'balanced_accuracy_score': make_scorer(balanced_accuracy_score)
    }

    clf_svc = make_pipeline(SVC(probability=True, class_weight='balanced'))
    cv_svc = RepeatedStratifiedKFold(n_splits=2, n_repeats=5,
                                     random_state=0)
    scores = cross_validate(clf_svc, X.copy(), y.copy(), cv=cv_svc, n_jobs=-1, scoring=scoring, return_estimator=True)


    # scores["roc_auc_score_mean"] = np.mean(scores["test_roc_auc_score"])
    a, b = make_roc_curve("out_dir", clf_svc, X.copy(), y.copy(), cv_svc, "param_str")
    # auc_cv = []
    # for e in scores['estimator']:
    #     y_pred = e.predict_proba(X)
    #     y_true = y
    #     aucs = []
    #     for fold in range(y_pred.shape[1]):
    #         y_p = y_pred[:, fold]
    #         fpr, tpr, _ = metrics.roc_curve(y_true, y_p, pos_label=1)
    #         aucs.append(metrics.auc(fpr, tpr))
    #     auc_cv.append(np.mean(aucs))
    # auc_cv_mean = np.mean(auc_cv)
    exit(0)


    classifier = make_pipeline(SVC(probability=True, class_weight='balanced'))
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=10,
                                     random_state=0)


    tprs = []
    fprs = []
    aucs = []
    bas = []
    for i, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X[train], y[train])
        # viz = plot_roc_curve(classifier, X[test], y[test],
        #                      label=None,
        #                      alpha=0.3, lw=1, ax=ax, c="tab:blue")
        # interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        # interp_tpr[0] = 0.0
        # tprs.append(interp_tpr)
        # aucs.append(viz.roc_auc)
        # ax.plot(viz.fpr, viz.tpr, c="tab:green")
        y_true = y[test]
        y_pred = classifier.predict(X[test])

        fpr, tpr, _ = metrics.roc_curve(y_true, y_pred, pos_label=1)
        tprs.append(tpr)
        fprs.append(fpr)
        aucs.append(metrics.auc(fpr, tpr))
        bas.append(balanced_accuracy_score(y_true, y_pred))
        p_s = precision_score(y_true, y_pred, average=None)
        p_s_0 = p_s[0]
        p_s_1 = p_s[1]

        r_s = recall_score(y_true, y_pred, average=None)
        r_s_0 = r_s[0]
        r_s_1 = r_s[1]



