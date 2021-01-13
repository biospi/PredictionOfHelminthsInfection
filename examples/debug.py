from sklearn import datasets, linear_model
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.metrics import recall_score, balanced_accuracy_score, roc_auc_score, precision_score, f1_score, roc_curve
import numpy as np

if __name__ == "__main__":
    diabetes = datasets.load_diabetes()
    X = diabetes.data[:150]
    # y = diabetes.target[:150]
    y = (np.random.rand(150) > 0.5).astype(int)


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



