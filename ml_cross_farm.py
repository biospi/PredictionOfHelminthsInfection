import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from model.data_loader import loadActivityData, parse_param_from_filename
from model.svm import makeRocCurve
from preprocessing.preprocessing import applyPreprocessingSteps
from utils.Utils import create_rec_dir
import scikitplot as skplt


def find_dataset(folder):
    files = glob.glob(folder + "/*.csv")  # find datset files
    files = [file.replace("\\", '/') for file in files]
    print("found %d files." % len(files))
    print(files)
    return files[0]


def main(farm1_path, farm2_path, class_healthy, class_unhealthy, output_dir, steps, n_process):
    print("farm_1=", farm1_path)
    print("farm_2=", farm2_path)
    print("n_process=", n_process)
    days, farm_id, option, sampling = parse_param_from_filename(farm1_path)
    dataset1, N_META, class_healthy_target, class_unhealthy_target, label_series = loadActivityData(find_dataset(farm1_path),
                                                                                                    days, class_healthy, class_unhealthy)
    dataset2, _, _, _, _ = loadActivityData(find_dataset(farm2_path), days, class_healthy, class_unhealthy)

    print(dataset1)
    print(dataset2)

    dataframe = pd.concat([dataset1, dataset2], axis=0)
    df_processed = applyPreprocessingSteps(days, None, None, None, None, None,
                                           dataframe.copy(), N_META, output_dir, steps,
                                           class_healthy_label, class_unhealthy_label, class_healthy_target,
                                           class_unhealthy_target, clf_name="SVM", output_dim=dataset1.shape[0],
                                           n_scales=None, farm_name="FARMS")

    df1_processed = df_processed.iloc[0:dataset1.shape[0], :]
    df2_processed = df_processed.iloc[dataset1.shape[0]:, :]

    # df1_processed = applyPreprocessingSteps(days, None, None, None, None, None,
    #                                        dataset1.copy(), N_META, output_dir, steps,
    #                                        class_healthy_label, class_unhealthy_label, class_healthy_target,
    #                                        class_unhealthy_target, clf_name="SVM", output_dim=dataset1.shape[0],
    #                                        n_scales=None, farm_name="FARM1")
    #
    # df2_processed = applyPreprocessingSteps(days, None, None, None, None, None,
    #                                        dataset2.copy(), N_META, output_dir, steps,
    #                                        class_healthy_label, class_unhealthy_label, class_healthy_target,
    #                                        class_unhealthy_target, clf_name="SVM", output_dim=dataset2.shape[0],
    #                                        n_scales=None, farm_name="FARM2")
    # print(df1_processed)
    # print(df2_processed)

    data_frame1 = df1_processed.loc[df1_processed['target'].isin([class_healthy_target, class_unhealthy_target])]
    data_frame2 = df2_processed.loc[df2_processed['target'].isin([class_healthy_target, class_unhealthy_target])]

    X1, y1 = getXY(data_frame1)
    X2, y2 = getXY(data_frame2)

    slug = "_".join(steps)
    processSVM(X1, X2, y1, y2, output_dir, slug, days)


def getXY(data_frame):
    y = data_frame['target'].values.flatten()
    y = y.astype(int)
    X = data_frame[data_frame.columns[0:data_frame.shape[1] - 1]].values
    y_binary = (y.copy() != 1).astype(int)

    return X, y_binary


def processSVM(X_train, X_test, y_train, y_test, output_dir, steps, days):
    clf_svc = SVC(kernel="rbf", probability=True, class_weight='balanced')
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': ['scale', 1e-1, 1e-3, 1e-4], 'class_weight': [None, 'balanced'],
                         'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    clf = GridSearchCV(clf_svc, tuned_parameters, scoring='roc_auc', n_jobs=-1)
    clf.fit(X_train.copy(), y_train.copy())
    clf_best = clf.best_estimator_
    print("Best estimator from gridsearch=")
    print(clf_best)
    y_pred = clf.predict(X_test.copy())
    print(classification_report(y_test, y_pred))

    filename = "%s/report.csv" % output_dir
    create_rec_dir(filename)
    print(filename)
    df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True))
    df.to_csv(filename)
    #makeRoc(clf_best, X_test, y_test, output_dir)
    y_proba = clf.predict_proba(X_test.copy())
    save_roc_curve(y_test, y_proba, output_dir)


def save_roc_curve(y_test, y_probas, out_dir):
    title = 'ROC Curve'
    skplt.metrics.plot_roc(y_test, y_probas, title=title, title_fontsize='medium')
    final_path = '%s/%s' % (out_dir, 'roc.png')
    create_rec_dir(final_path)
    print(final_path)
    plt.savefig(final_path)
    plt.show()
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("farm_1", help='Dataset input directory', type=str)
    parser.add_argument("farm_2", help='Dataset input directory', type=str)
    parser.add_argument("output_dir", help="Output directory", type=str)
    parser.add_argument("--class_healthy_label", help="Label for healthy class", type=str, default="1To1")
    parser.add_argument("--class_unhealthy_label", help="Label for unhealthy class", type=str, default="2To2")
    parser.add_argument('--n_process', help='Number of threads to use.', default=6, type=int)

    args = parser.parse_args()
    farm_1 = args.farm_1
    farm_2 = args.farm_2
    output_dir = args.output_dir
    class_healthy_label = args.class_healthy_label
    class_unhealthy_label = args.class_unhealthy_label
    n_process = args.n_process

    steps = ["QN", "ANSCOMBE", "LOG", "DIFF"]

    main(farm_1, farm_2, class_healthy_label, class_unhealthy_label, output_dir, steps, n_process)