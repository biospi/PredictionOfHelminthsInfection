import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Manager, Pool
import time
import json
import pickle
from keras import metrics
from sklearn.metrics import plot_roc_curve

from utils.visualisation import plot_roc_range
from sklearn.metrics import recall_score, balanced_accuracy_score, precision_score, f1_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_curve, classification_report
from sklearn.metrics import auc
import warnings
from pathlib import Path


def plot_model_metrics(history, out_dir, i, dir_name="model_cnn"):
    # summarize history for accuracy
    fig_fold, ax_fold = plt.subplots()
    ax_fold.plot(history.history['auc'])
    ##plt.plot(history.history['val_auc'])
    ax_fold.set_title('model auc for fold %d' % i)
    ax_fold.set_ylabel('auc')
    ax_fold.set_xlabel('epoch')
    ax_fold.legend(['train', 'test'], loc='upper left')
    path = "%s/%s" % (out_dir, dir_name)
    Path(path).mkdir(parents=True, exist_ok=True)
    final_path = '%s/%s' % (path, 'fold%d_modelauc.png' % (i))
    print(final_path)
    fig_fold.savefig(final_path)
    fig_fold.clear()
    plt.close(fig_fold)

    fig_fold, ax_fold = plt.subplots()
    # summarize history for loss
    ax_fold.plot(history.history['loss'])
    ##plt.plot(history.history['val_loss'])
    ax_fold.set_title('model loss for fold %d' % i)
    ax_fold.set_ylabel('loss')
    ax_fold.set_xlabel('epoch')
    ax_fold.legend(['train', 'test'], loc='upper left')
    path = "%s/%s" % (out_dir, dir_name)
    Path(path).mkdir(parents=True, exist_ok=True)
    final_path = '%s/%s' % (path, 'fold%d_modelloss.png' % (i))
    print(final_path)
    fig_fold.savefig(final_path)
    fig_fold.clear()
    plt.close(fig_fold)


def plot_roc(ax, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_test = np.argmax(y_test, axis=1)
    y_pred_idx = np.argmax(y_pred, axis=1)
    y_pred_proba = y_pred[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    ax.plot(fpr, tpr, label=None, alpha=0.3, lw=1, c="tab:blue")
    roc_auc = auc(fpr, tpr)
    report = classification_report(y_test, y_pred_idx, output_dict=True)
    print(report)
    targets = list(report.keys())[:-3]
    if len(targets) == 1:
        warnings.warn("classifier only predicted 1 target.")
    #     if targets[0] == "0":
    #         report["1"] = {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0}
    #     if targets[0] == "1":
    #         report["0"] = {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0}
    # print(report)
    # if np.isnan(roc_auc):
    #     roc_auc = 0
    print("roc_auc=", roc_auc)
    p0, p1, r0, r1, f0, f1 = 0, 0, 0, 0, 0, 0
    acc = report["accuracy"]
    if "0" in report:
        p0 = report["0"]["precision"]
        r0 = report["0"]["recall"]
        f0 = report["0"]["f1-score"]

    if "1" in report:
        p1 = report["1"]["precision"]
        r1 = report["1"]["recall"]
        f1 = report["1"]["f1-score"]

    return roc_auc, fpr, tpr, report, acc, p0, p1, r0, r1, f0, f1


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res


def build_model(
    n_classes,
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def fold_worker(
    out_dir,
    y_h,
    ids,
    meta,
    meta_data_short,
    sample_dates,
    days,
    steps,
    tprs_test,
    tprs_train,
    aucs_roc_test,
    aucs_roc_train,
    fold_results,
    fold_probas,
    label_series,
    mean_fpr_test,
    mean_fpr_train,
    clf_name,
    X,
    y,
    train_index,
    test_index,
    axis_test,
    axis_train,
    ifold,
    nfold,
    epochs,
    batch_size,
):
    print(f"process id={ifold}/{nfold}...")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    y_h_train, y_h_test = y_h[train_index], y_h[test_index]
    ids_train, ids_test = ids[train_index], ids[test_index]

    meta_train, meta_test = meta[train_index], meta[test_index]
    meta_train_s, meta_test_s = (
        meta_data_short[train_index],
        meta_data_short[test_index],
    )

    sample_dates_train, sample_dates_test = (
        sample_dates[train_index],
        sample_dates[test_index],
    )
    class_healthy, class_unhealthy = 0, 1

    # hold all extra label
    fold_index = np.array(train_index.tolist() + test_index.tolist())
    X_fold = X[fold_index]
    y_fold = y[fold_index]
    ids_fold = ids[fold_index]
    sample_dates_fold = sample_dates[fold_index]
    meta_fold = meta[fold_index]

    # keep healthy and unhealthy only
    X_train = X_train[np.isin(y_h_train, [class_healthy, class_unhealthy])]
    y_train = y_h_train[np.isin(y_h_train, [class_healthy, class_unhealthy])]
    meta_train = meta_train[np.isin(y_h_train, [class_healthy, class_unhealthy])]
    meta_train_s = meta_train_s[np.isin(y_h_train, [class_healthy, class_unhealthy])]

    X_test = X_test[np.isin(y_h_test, [class_healthy, class_unhealthy])]
    y_test = y_h_test[np.isin(y_h_test, [class_healthy, class_unhealthy])]

    meta_test = meta_test[np.isin(y_h_test, [class_healthy, class_unhealthy])]
    meta_test_s = meta_test_s[np.isin(y_h_test, [class_healthy, class_unhealthy])]
    ids_test = ids_test[np.isin(y_h_test, [class_healthy, class_unhealthy])]
    sample_dates_test = sample_dates_test[
        np.isin(y_h_test, [class_healthy, class_unhealthy])
    ]

    ids_train = ids_train[np.isin(y_h_train, [class_healthy, class_unhealthy])]
    sample_dates_train = sample_dates_train[
        np.isin(y_h_train, [class_healthy, class_unhealthy])
    ]

    start_time = time.time()
####################################################
    #classes = np.unique(np.concatenate((y_train, y_test), axis=0))

    # plt.figure()
    # for c in classes:
    #     c_x_train = X_train[y_train == c]
    #     plt.plot(c_x_train[0], label="class " + str(c))
    # plt.legend(loc="best")
    # plt.show()
    # plt.close()

    x_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1)).astype('float32')
    x_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1)).astype('float32')

    num_classes = len(np.unique(y_train))

    idx = np.random.permutation(len(x_train))
    x_train = x_train[idx]
    y_train = y_train[idx]

    input_shape = x_train.shape[1:]
    model = build_model(
        num_classes,
        input_shape,
        head_size=256,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=[128],
        mlp_dropout=0.4,
        dropout=0.25,
    )

    keras.utils.plot_model(
        model, to_file=out_dir / 'cnn_model.png', show_shapes=False, show_dtype=False,
        show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96
    )

##############################################################
    METRICS = [
        metrics.TruePositives(name='tp'),
        metrics.FalsePositives(name='fp'),
        metrics.TrueNegatives(name='tn'),
        metrics.FalseNegatives(name='fn'),
        metrics.BinaryAccuracy(name='accuracy'),
        metrics.Precision(name='precision'),
        metrics.Recall(name='recall'),
        metrics.AUC(name='auc')
    ]

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["sparse_categorical_accuracy"],
    )

    model.summary()

    model_dir = out_dir / "cnn_models"
    print(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            model_dir / f"best_model_{ifold}.h5", save_best_only=True, monitor="val_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
    ]

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_split=0.2,
        verbose=1,
    )

    fit_time = time.time() - start_time

    #plot_model_metrics(history, out_dir, ifold, dir_name="model_1dcnn")

###############################################################
    model = keras.models.load_model(out_dir / "cnn_models" / f"best_model_{ifold}.h5")

    test_loss, test_acc = model.evaluate(x_test, y_test)

    print("Test accuracy", test_acc)
    print("Test loss", test_loss)

    # metric = "sparse_categorical_accuracy"
    # plt.figure()
    # plt.plot(history.history[metric])
    # plt.plot(history.history["val_" + metric])
    # plt.title("model " + metric)
    # plt.ylabel(metric, fontsize="large")
    # plt.xlabel("epoch", fontsize="large")
    # plt.legend(["train", "val"], loc="best")
    # plt.show()
    # plt.close()

    #roc_auc, fpr, tpr, report, acc, p0, p1, r0, r1, f0, f1 = plot_roc(axis_test, model, X_test, y_test)

###############################################################
    # clf.fit(X_train, y_train)
    # fit_time = time.time() - start_time
    #
    # models_dir = (
    #     out_dir / "models" / f"{clf_name}_{days}_{steps}"
    # )
    #
    # models_dir.mkdir(parents=True, exist_ok=True)
    # filename = models_dir / f"model_{ifold}.pkl"
    # print("saving classifier...")
    # print(filename)
    # with open(str(filename), "wb") as f:
    #     pickle.dump(clf, f)

    # test healthy/unhealthy
    y_pred = model.predict(X_test)
    y_pred_proba_test = y_pred
    y_pred = (y_pred[:, 1] >= 0.5).astype(int)

    # prep for roc curve
    # viz_roc_test = plot_roc_curve(
    #     clf,
    #     X_test,
    #     y_test,
    #     label=None,
    #     alpha=0.3,
    #     lw=1,
    #     ax=None,
    #     c="tab:blue",
    # )
    #axis_test.append(viz_roc_test)

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba_test[:, 1])
    axis_test.append({"fpr": fpr, "tpr": tpr})

    interp_tpr_test = np.interp(mean_fpr_test, fpr, tpr)
    interp_tpr_test[0] = 0.0
    tprs_test.append(interp_tpr_test)
    auc_value_test = auc(fpr, tpr)
    print("auc test=", auc_value_test)
    aucs_roc_test.append(auc_value_test)

    # viz_roc_train = plot_roc_curve(
    #     clf,
    #     X_train,
    #     y_train,
    #     label=None,
    #     alpha=0.3,
    #     lw=1,
    #     ax=None,
    #     c="tab:blue",
    # )
    # axis_train.append(viz_roc_train)

    y_pred_train = model.predict(X_train)
    y_pred_proba_train = y_pred_train
    y_pred_train = (y_pred_train[:, 1] >= 0.5).astype(int)

    fpr, tpr, _ = roc_curve(y_train, y_pred_proba_train[:, 1])
    axis_train.append({"fpr": fpr, "tpr": tpr})

    interp_tpr_train = np.interp(mean_fpr_train, fpr, tpr)
    interp_tpr_train[0] = 0.0
    tprs_train.append(interp_tpr_train)
    auc_value_train = auc(fpr, tpr)
    print("auc train=", auc_value_train)
    aucs_roc_train.append(auc_value_train)

    # if ifold == 0:
    #     plot_high_dimension_db(
    #         out_dir / "testing",
    #         np.concatenate((X_train, X_test), axis=0),
    #         np.concatenate((y_train, y_test), axis=0),
    #         list(np.arange(len(X_train))),
    #         np.concatenate((meta_train_s, meta_test_s), axis=0),
    #         clf,
    #         days,
    #         steps,
    #         ifold,
    #     )
    #     plot_learning_curves(clf, X, y, ifold, out_dir / "testing")

    accuracy = balanced_accuracy_score(y_test, y_pred)
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred)
    correct_predictions_test = (y_test == y_pred).astype(int)
    incorrect_predictions_test = (y_test != y_pred).astype(int)

    # data for training
    accuracy_train = balanced_accuracy_score(y_train, y_pred_train)
    (
        precision_train,
        recall_train,
        fscore_train,
        support_train,
    ) = precision_recall_fscore_support(y_train, y_pred_train)
    correct_predictions_train = (y_train == y_pred_train).astype(int)
    incorrect_predictions_train = (y_train != y_pred_train).astype(int)

    fold_result = {
        "target": int(class_unhealthy),
        "auc": auc_value_test,
        "accuracy": float(accuracy),
        "accuracy_train": float(accuracy_train),
        "class_healthy": int(class_healthy),
        "class_unhealthy": int(class_unhealthy),
        "y_test": y_test.tolist(),
        "y_pred_proba_test": y_pred_proba_test.tolist(),
        "y_pred_proba_train": y_pred_proba_train.tolist(),
        "ids_test": ids_test.tolist(),
        "ids_train": ids_train.tolist(),
        "sample_dates_test": sample_dates_test.tolist(),
        "sample_dates_train": sample_dates_train.tolist(),
        "meta_test": meta_test.tolist(),
        "meta_fold": meta_fold.tolist(),
        "meta_train": meta_train.tolist(),
        "correct_predictions_test": correct_predictions_test.tolist(),
        "incorrect_predictions_test": incorrect_predictions_test.tolist(),
        "correct_predictions_train": correct_predictions_train.tolist(),
        "incorrect_predictions_train": incorrect_predictions_train.tolist(),
        "test_precision_score_0": float(precision[0]),
        "test_precision_score_1": float(precision[1]),
        "test_recall_0": float(recall[0]),
        "test_recall_1": float(recall[1]),
        "test_fscore_0": float(fscore[0]),
        "test_fscore_1": float(fscore[1]),
        "test_support_0": float(support[0]),
        "test_support_1": float(support[1]),
        "train_precision_score_0": float(precision_train[0]),
        "train_precision_score_1": float(precision_train[1]),
        "train_recall_0": float(recall_train[0]),
        "train_recall_1": float(recall_train[1]),
        "train_fscore_0": float(fscore_train[0]),
        "train_fscore_1": float(fscore_train[1]),
        "train_support_0": float(support_train[0]),
        "train_support_1": float(support_train[1]),
        "fit_time": fit_time,
    }
    fold_results.append(fold_result)

    # test individual labels and store probabilities to be healthy/unhealthy
    for y_f in y_fold:
        label = label_series[y_f]
        X_test = X_fold[y_fold == y_f]
        y_test = y_fold[y_fold == y_f]
        y_pred_proba_test = model.predict(X_test)
        fold_proba = {
            "test_y_pred_proba_0": y_pred_proba_test[:, 0].tolist(),
            "test_y_pred_proba_1": y_pred_proba_test[:, 1].tolist(),
        }
        fold_probas[label].append(fold_proba)
    print(f"process id={ifold}/{nfold} done!")


def cross_validate_cnn(
    svc_kernel,
    out_dir,
    steps,
    cv_name,
    days,
    label_series,
    cross_validation_method,
    X,
    y,
    y_h,
    ids,
    meta,
    meta_data_short,
    sample_dates,
    clf_name,
    n_job,
    epochs=500,
    batch_size=32
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
    scores, scores_proba = {}, {}
    plt.clf()
    fig_roc, ax_roc = plt.subplots(1, 2, figsize=(19.20, 6.20))
    fig_roc_merge, ax_roc_merge = plt.subplots(figsize=(12.80, 7.20))
    mean_fpr_test = np.linspace(0, 1, 100)
    mean_fpr_train = np.linspace(0, 1, 100)

    with Manager() as manager:
        # create result holders
        tprs_test = manager.list()
        tprs_train = manager.list()
        axis_test = manager.list()
        axis_train = manager.list()
        aucs_roc_test = manager.list()
        aucs_roc_train = manager.list()
        fold_results = manager.list()
        fold_probas = manager.dict()
        for k in label_series.values():
            fold_probas[k] = manager.list()

        pool = Pool(processes=n_job)
        start = time.time()
        for ifold, (train_index, test_index) in enumerate(
            cross_validation_method.split(X, y)
        ):
            pool.apply_async(
                fold_worker,
                (
                    out_dir,
                    y_h,
                    ids,
                    meta,
                    meta_data_short,
                    sample_dates,
                    days,
                    steps,
                    tprs_test,
                    tprs_train,
                    aucs_roc_test,
                    aucs_roc_train,
                    fold_results,
                    fold_probas,
                    label_series,
                    mean_fpr_test,
                    mean_fpr_train,
                    clf_name,
                    X,
                    y,
                    train_index,
                    test_index,
                    axis_test,
                    axis_train,
                    ifold,
                    cross_validation_method.get_n_splits(),
                    epochs,
                    batch_size
                ),
            )
        pool.close()
        pool.join()
        end = time.time()
        fold_results = list(fold_results)
        axis_test = list(axis_test)
        tprs_test = list(tprs_test)
        aucs_roc_test = list(aucs_roc_test)
        axis_train = list(axis_train)
        tprs_train = list(tprs_train)
        aucs_roc_train = list(aucs_roc_train)
        fold_probas = dict(fold_probas)
        fold_probas = dict([a, list(x)] for a, x in fold_probas.items())
        print("total time (s)= " + str(end - start))

    info = f"X shape:{str(X.shape)} healthy:{np.sum(y_h == 0)} unhealthy:{np.sum(y_h == 1)}"
    for a in axis_test:
        xdata = a["fpr"]
        ydata = a["tpr"]
        ax_roc[1].plot(xdata, ydata, color="tab:blue", alpha=0.3, linewidth=1)
        ax_roc_merge.plot(xdata, ydata, color="tab:blue", alpha=0.3, linewidth=1)

    for a in axis_train:
        xdata = a["fpr"]
        ydata = a["tpr"]
        ax_roc[0].plot(xdata, ydata, color="tab:blue", alpha=0.3, linewidth=1)
        ax_roc_merge.plot(xdata, ydata, color="tab:purple", alpha=0.3, linewidth=1)

    mean_auc = plot_roc_range(
        ax_roc_merge,
        ax_roc,
        tprs_test,
        mean_fpr_test,
        aucs_roc_test,
        tprs_train,
        mean_fpr_train,
        aucs_roc_train,
        out_dir,
        steps,
        fig_roc,
        fig_roc_merge,
        cv_name,
        days,
        info=info,
        tag=f"{clf_name}",
    )

    scores[f"{clf_name}_results"] = fold_results
    scores_proba[f"{clf_name}_probas"] = fold_probas

    print("export results to json...")
    filepath = out_dir / "results.json"
    print(filepath)
    with open(str(filepath), "w") as fp:
        json.dump(scores, fp)
    filepath = out_dir / "results_proba.json"
    print(filepath)
    with open(str(filepath), "w") as fp:
        json.dump(scores_proba, fp)

    return scores, scores_proba