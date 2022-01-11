import pickle
from typing import List

import typer
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report
import matplotlib.pyplot as plt
from model.data_loader import parse_param_from_filename
from preprocessing.preprocessing import apply_preprocessing_steps
import scikitplot as skplt


def load_activity_data(filepath, imputed_days=6):
    print(f"load activity from datasets...{filepath}")
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
    data_frame = data_frame[data_frame["imputed_days"] <= imputed_days]
    data_frame = data_frame.dropna()
    data_frame = data_frame.fillna(-1)

    data_frame_labeled = pd.get_dummies(data_frame, columns=["label"])
    flabels = [x for x in data_frame_labeled.columns if 'label' in x]
    data_frame["target"] = 0
    for i, flabel in enumerate(flabels):
        data_frame_labeled[flabel] = data_frame_labeled[flabel] * (i + 1)
        data_frame["target"] = data_frame["target"] + data_frame_labeled[flabel]

    labels = data_frame["label"].drop_duplicates().values

    samples = {}
    for label in labels:
        df = data_frame[data_frame["label"] == label]
        df = df.drop('label', 1)
        samples[label] = df

    return samples, N_META


def main(
    output_dir: Path = typer.Option(..., exists=False, file_okay=False, dir_okay=True, resolve_path=True),
    dataset_folder: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    model_file: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False, resolve_path=True),
    imputed_days: int = 6,
    preprocessing_steps: List[str] = ["QN", "ANSCOMBE", "LOG"],
):
    """Evaluated trained model\n
    Args:\n
        output_dir: Output directory
        dataset_folder: Dataset input directory
        model_file: Fitted model location
    """
    files = [str(x) for x in list(dataset_folder.glob('*.csv'))] # find datset files
    print("found %d files." % len(files))
    print(files)
    for file in files:
        days, farm_id, option, sampling = parse_param_from_filename(file)

    samples, N_META = load_activity_data(files[0], imputed_days)

    with open(str(model_file), 'rb') as f:
        clf = pickle.load(f)

    predict_list = []
    test_labels = []
    test_size = []
    for test_label, X in samples.items():
        df_processed = apply_preprocessing_steps(
            days,
            None,
            None,
            None,
            None,
            [],
            X.copy(),
            N_META,
            output_dir / test_label,
            preprocessing_steps,
            "class_healthy_label",
            "class_unhealthy_label",
            1,
            2,
            clf_name="SVM",
            output_dim=X.shape[0],
            n_scales=None,
        )

        X_test = df_processed.iloc[:, :-1].values
        y_test = df_processed["target"].values
        #predict = clf.predict(X_test)
        predict = clf.predict_proba(X_test)[:, 1]
        predict_list.append(predict)
        test_labels.append(test_label)
        test_size.append(X_test.shape[0])

        print(predict)
        plt.xlabel("Probability to be unhealthy(2To2)", size=14)
        plt.ylabel("Count", size=14)
        plt.hist(predict, bins=30)
        plt.title(f"Histogram of predictions test_label={test_label} test_size={X_test.shape[0]}")

        filename = f"{test_label}_{model_file.stem}.png"
        out = output_dir / filename
        print(out)
        plt.savefig(str(out))

        plt.show()
        plt.close()


        # y_test = np.zeros(df_processed.shape[0])
        #
        # y_pred_proba = clf.predict_proba(X_test)[::, 1]
        # fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        # auc = roc_auc_score(y_test, y_pred_proba)
        # plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
        # plt.legend(loc=4)
        # plt.show()
        #
        # filename = f"ROC_{test_label}_{model_file.stem}.png"
        # out = output_dir / filename
        # print(out)
        # plt.savefig(str(out))

        #
        # y_test[:] = 1
        #
        # probs = clf.predict_proba(X_test)
        #
        # preds = probs[:, 1]
        # fpr, tpr, threshold = roc_curve(y_test, preds)
        # roc_auc = auc(fpr, tpr)
        # plt.title('Receiver Operating Characteristic')
        # plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        # plt.legend(loc='lower right')
        # plt.plot([0, 1], [0, 1], 'r--')
        # plt.xlim([0, 1])
        # plt.ylim([0, 1])
        # plt.ylabel('True Positive Rate')
        # plt.xlabel('False Positive Rate')
        # plt.show()
        # filename = f"ROC_{test_label}_{model_file.stem}.png"
        # out = output_dir / filename
        # print(out)
        # plt.savefig(str(out))

    print(predict_list)

    for bin_size in [5, 10, 30, 50, 100]:
        plt.clf()
        test_label_str = ''
        test_size_str = ''
        for data, label, size in zip(predict_list, test_labels, test_size):
            plt.hist(data, bins=bin_size, alpha=0.5, label=f'{label}({str(size)})')
            test_label_str += label + ','
            test_size_str += str(size) + ','
        plt.xlabel("Probability to be unhealthy(2To2)", size=14)
        plt.ylabel("Count", size=14)
        plt.title(f"Histogram of predictions (bin size={bin_size})\n test_label={test_label_str[:-1]} test_size={test_size_str[:-1]}")
        plt.legend(loc='upper right')
        filename = f"binsize_{bin_size}_{test_label_str.replace(',','_')}_{model_file.stem}.png"
        out = output_dir / filename
        print(out)
        plt.savefig(str(out))


if __name__ == "__main__":
    #typer.run(main)
    main(Path(f"E:/Data2/mrnn_median_ml_debug/"), Path("E:/Data2/mrnn_datasets2/median_10080_100_all/dataset_gain_7day"),
         Path("E:/Data2/mrnn_median_ml_all/10080_100_2To2_7/RepeatedKFold/model_7_QN_ANSCOMBE_LOG_linear.pkl"))