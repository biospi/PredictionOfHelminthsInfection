from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold, LeaveOneOut

from cnn.cnn import run1DCnn, run2DCnn
from model.svm import downsampleDf
from utils._custom_split import StratifiedLeaveTwoOut


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

