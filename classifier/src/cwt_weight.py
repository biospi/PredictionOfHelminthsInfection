import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from sklearn.utils import shuffle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pathlib
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import scale
import seaborn as sns
from sklearn.preprocessing import normalize
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
import json
import eli5
import pycwt as wavelet
from pycwt.helpers import find
from scipy.signal import chirp

L = 0
F = None
def create_cwt_graph(coefs, freq, lenght, title=None):
    # time = [x for x in range(0, lenght)]
    plt.matshow(coefs)
    # plt.show()

    # fig = plt.figure()
    # plt.matshow(coefs.real)
    # # plt.pcolormesh(time, freq, coef)
    # fig.suptitle(title, x=0.5, y=.95, horizontalalignment='center', verticalalignment='top', fontsize=10)
    # path = "training_sets_cwt_graphs"
    # pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    # # fig.savefig('%s/%s.png' % (path, title))
    # fig.show()
    # exit()


def interpolate(input_activity):
    try:
        i = np.array(input_activity, dtype=np.float)
        s = pd.Series(i)
        s = s.interpolate(method='cubic', limit_direction='both')
        s = s.interpolate(method='linear', limit_direction='both')
        return s.tolist()
    except ValueError as e:
        print(e)
        return input_activity


def even_list(n):
    result = [1]
    for num in range(2, n * 2 + 1, 2):
        result.append(num)
    del result[-1]
    return np.asarray(result, dtype=np.int32)


def dummy_sin():
    T = 5
    n = 1000
    t = np.linspace(0, T, n, endpoint=False)
    f0 = 1
    f1 = 10
    y = chirp(t, f0, T, f1, method='logarithmic')

    plt.plot(t, y)
    plt.grid(alpha=0.25)
    plt.xlabel('t (seconds)')
    plt.show()
    return t, y


def compute_cwt(activity):
    print("compute_cwt...")
    # t, activity = dummy_sin()
    num_steps = len(activity)
    x = np.arange(num_steps)
    y = activity

    delta_t = x[1] - x[0]
    scales = np.arange(1, num_steps + 1)/1
    freqs = 1 / (wavelet.Morlet().flambda() * scales)
    wavelet_type = 'morlet'

    coefs, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(y, delta_t, wavelet=wavelet_type, freqs=freqs)

    # iwave = wavelet.icwt(coefs, scales, delta_t, wavelet=wavelet_type)
    # plt.plot(iwave)
    # plt.show()
    # plt.plot(activity)
    # plt.show()
    #
    # plt.matshow(coefs.real)
    # plt.show()
    # exit()
    cwt = [element for tupl in coefs.real for element in tupl]
    indexes = list(range(len(cwt)))
    indexes.reverse()
    return cwt, coefs.real, freqs, indexes, scales, delta_t, wavelet_type


def start(fname=''):
    print("loading dataset...")
    # print(fname)
    data_frame = pd.read_csv(fname, sep=",", header=None)
    # print(data_frame)
    sample_count = data_frame.shape[1]
    hearder = [str(n) for n in range(0, sample_count)]
    hearder[-5] = "class"
    hearder[-4] = "elem_in_row"
    hearder[-3] = "date1"
    hearder[-2] = "date2"
    hearder[-1] = "serial"
    data_frame.columns = hearder
    data_frame = data_frame.loc[:, :'class']
    np.random.seed(0)
    data_frame = data_frame.sample(frac=1).reset_index(drop=True)
    data_frame = data_frame.fillna(-1)
    data_frame = shuffle(data_frame)
    data_frame['class'] = data_frame['class'].map({True: 1, False: 0})
    process(data_frame)


data = []
def process_data_frame(data_frame):
    data_frame = data_frame.fillna(-1)
    X = data_frame[data_frame.columns[0:data_frame.shape[1] - 1]].values
    cwt_list = []
    class0 = []
    class1 = []
    for i, activity in enumerate(X):
        print("%d/%d ..." % (i, len(X)))
        cwt, coefs, freqs, indexes, scales, delta_t, wavelet_type = compute_cwt(activity)
        data.append({'coef': coefs, 'freqs': freqs, 'shape': activity.shape[0], 'title': str(i)})
        # create_cwt_graph(coefs, freqs, activity.shape[0], title=str(i))
        cwt_list.append(cwt)
        target = data_frame.at[i, 'class']
        if target == 0:
            class0.append(activity)
        if target == 1:
            class1.append(activity)

    class0_mean = np.average(class0, axis=0)
    class1_mean = np.average(class1, axis=0)

    # plt.plot(class0_mean)
    # plt.show()
    # plt.plot(class1_mean)
    # plt.show()

    _, coefs_class0_mean, _, _, _, _, _ = compute_cwt(class0_mean)
    # plt.matshow(coefs_class0_mean)
    # plt.show()
    _, coefs_class1_mean, _, _, _, _, _ = compute_cwt(class1_mean)
    # plt.matshow(coefs_class1_mean)
    # plt.show()
    #
    # exit()

    X = pd.DataFrame.from_records(cwt_list)

    y = data_frame["class"].values.flatten()
    # X = normalize(X)
    # X = preprocessing.MinMaxScaler().fit_transform(X)
    return X.values, y, scales, delta_t, wavelet_type, class0_mean, coefs_class0_mean, class1_mean, coefs_class1_mean


def plot_coefficients(classifier, feature_names, top_features=20):
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15, 5))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
    plt.show()


def class_feature_importance(X, Y, feature_importances):
    N, M = X.shape
    X = scale(X)

    out = {}
    for c in set(Y):
        out[c] = dict(
            zip(range(N), np.mean(X[Y==c, :], axis=0)*feature_importances)
        )

    return out


def pad(A, length):
    arr = np.zeros(length)
    arr[:len(A)] = A
    return arr


def normalized(v):
    return v / np.sqrt(np.sum(v**2))


def process(data_frame):
    print("process...")
    X, y, scales, delta_t, wavelet_type, class0_mean, coefs_class0_mean, class1_mean, coefs_class1_mean = process_data_frame(data_frame)
    print("train_test_split...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    clf = SVC(kernel='linear')
    # clf = LDA(n_components=2)

    y_train[0] = 3
    print("fit...")
    clf.fit(X_train, y_train)

    # result = class_feature_importance(X.values, y, clf.coef_[0])
    print("explain_prediction...")
    aux1 = eli5.sklearn.explain_prediction.explain_prediction_linear_classifier(clf, X[0], top=X.shape[1])

    aux1 = eli5.format_as_dataframe(aux1)

    class0 = aux1[aux1.target == 0]
    class1 = aux1[aux1.target == 1]

    class0 = class0[class0.feature != '<BIAS>']
    class1 = class1[class1.feature != '<BIAS>']

    class0['feature'] = class0['feature'].str.replace('x', '')
    class1['feature'] = class1['feature'].str.replace('x', '')

    class0['feature'] = class0['feature'].apply(int)
    class1['feature'] = class1['feature'].apply(int)

    class0 = class0.sort_values('feature')
    class1 = class1.sort_values('feature')

    value0 = class0['value'].values
    weight0 = class0['weight'].values

    value1 = class1['value'].values
    weight1 = class1['weight'].values

    value0 = pad(value0, data[0]['coef'].shape[0] * data[0]['coef'].shape[1])
    weight0 = pad(weight0, data[0]['coef'].shape[0] * data[0]['coef'].shape[1])
    value1 = pad(value1, data[0]['coef'].shape[0] * data[0]['coef'].shape[1])
    weight1 = pad(weight1, data[0]['coef'].shape[0] * data[0]['coef'].shape[1])

    fig, axs = plt.subplots(6, 2)
    fig.set_size_inches(25, 25)


    c = np.reshape(value0, data[0]['coef'].shape)
    axs[5, 0].set_yscale('log')
    axs[5, 0].matshow(c)
    axs[5, 0].set_title('mean cwt input')
    iwave_m = wavelet.icwt(c, scales, delta_t, wavelet=wavelet_type)
    iwave_m = minmax_scale(np.real(iwave_m), feature_range=(-1, 1))

    axs[5, 1].plot(iwave_m)
    axs[5, 1].set_title('mean time input')

    axs[0, 0].set_yscale('log')
    axs[0, 0].matshow(coefs_class0_mean)
    axs[0, 0].set_title('class0 cwt input')
    iwave = wavelet.icwt(coefs_class0_mean, scales, delta_t, wavelet=wavelet_type)
    iwave = minmax_scale(np.real(iwave), feature_range=(-1, 1))
    axs[0, 1].plot(iwave)
    axs[0, 1].set_title('class0 time input')

    c = np.reshape(normalized(weight0), data[0]['coef'].shape)
    axs[1, 0].set_yscale('log')
    axs[1, 0].matshow(c)
    axs[1, 0].set_title('class0 cwt weight')
    iwave = wavelet.icwt(c, scales, delta_t, wavelet=wavelet_type)
    iwave = minmax_scale(np.real(iwave), feature_range=(-1, 1))
    axs[1, 1].plot(iwave)
    axs[1, 1].set_title('class0 cwt weight inverse')

    axs[2, 0].set_yscale('log')
    axs[2, 0].matshow(coefs_class1_mean)
    axs[2, 0].set_title('class1 cwt input')
    iwave = wavelet.icwt(coefs_class1_mean, scales, delta_t, wavelet=wavelet_type)
    iwave = minmax_scale(np.real(iwave), feature_range=(-1, 1))
    axs[2, 1].plot(iwave)
    axs[2, 1].set_title('class1 time input')
    c = np.reshape(normalized(weight1), data[0]['coef'].shape)

    axs[3, 0].set_yscale('log')
    axs[3, 0].matshow(c)
    axs[3, 0].set_title('class1 cwt weight')
    iwave = wavelet.icwt(c, scales, delta_t, wavelet=wavelet_type)
    iwave = minmax_scale(np.real(iwave), feature_range=(-1, 1))
    axs[3, 1].plot(iwave)
    axs[3, 1].set_title('class1 cwt weight inverse')

    c = np.reshape(coefs_class1_mean-coefs_class0_mean, data[0]['coef'].shape)
    axs[4, 0].set_yscale('log')
    axs[4, 0].matshow(c)
    axs[4, 0].set_title('diff')
    iwave = wavelet.icwt(c, scales, delta_t, wavelet=wavelet_type)
    iwave = minmax_scale(np.real(iwave), feature_range=(-1, 1))
    axs[4, 1].plot(iwave)
    axs[4, 1].set_title('diff')



    # for ax in axs.flat:
    #     ax.label_outer()

    fig.suptitle('SVC')

    fig.show()
    fig.savefig('SVC8.png', dpi=100)
    exit()

    # class_true_coef = list(result[True].values())
    # class_false_coef = list(result[False].values())
    # print(class_false_coef)
    # print(class_true_coef)

    # class_true_coef = np.reshape(class_true_coef, data[0]['coef'].shape)
    # class_false_coef = np.reshape(class_false_coef, data[0]['coef'].shape)

    # df_true = pd.DataFrame.from_records(class_true_coef)
    # df_true = normalize(df)
    # df_true = preprocessing.MinMaxScaler().fit_transform(df_true)
    # plt.pcolor(class_true_coef)
    # plt.show()

    # df_false = pd.DataFrame.from_records(class_false_coef)
    # df_true = normalize(df)
    # df_false = preprocessing.MinMaxScaler().fit_transform(df_false)
    # plt.pcolor(class_false_coef)
    # plt.show()

    # plot_coefficients(clf, features_name)
    y_pred = clf.predict(X_test)

    for i, item in enumerate(data):
        create_cwt_graph(item['coef'], item['freqs'], item['shape'], title=item['title'])

        # contributions = clf.coef_.reshape(item['coef'].flatten().shape)
        # contributions = np.reshape(contributions, item['coef'].shape)
        w = clf.coef_*1
        shape = item['coef'].shape
        contributions = np.reshape(w, shape)
        inverse_cwt(contributions)

        df = pd.DataFrame.from_records(contributions)

        df = normalize(df)
        # df = preprocessing.MinMaxScaler().fit_transform(df)

        plt.pcolor(df)
        # plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
        # plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
        plt.show()


        # create_cwt_graph(contributions, item['freqs'], item['shape'], title=item['title']+'_w')
        break

        # feature_number = np.arange(len(contributions)) + 1
        # plt.bar(feature_number, contributions, align='center')
        # plt.xlabel('feature index')
        # plt.ylabel('score contribution')
        # plt.title('contribution to classification outcome by feature index')
        # plt.show()


    acc = accuracy_score(y_test, y_pred)
    print(acc)


if __name__ == '__main__':
    # from sklearn import datasets
    # from sklearn.model_selection import train_test_split
    # from sklearn.tree import DecisionTreeClassifier
    # from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier,
    #                               AdaBoostClassifier, GradientBoostingClassifier)
    # import eli5
    #
    # iris = datasets.load_iris()  # sample data
    #compute_cwt
    #
    # X, y = iris.data[0:100], iris.target[0:100]
    # # split into training and test
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.33, random_state=0)
    #
    # # fit the model on the training set
    # # model = DecisionTreeClassifier(random_state=0)
    # model = LDA(n_components=2)
    #
    # model.fit(X_train, y_train)
    #
    # aux1 = eli5.sklearn.explain_prediction.explain_prediction_linear_classifier(model, X[0], top=X.shape[1])
    #
    # aux1 = eli5.format_as_dataframe(aux1)
    # print(aux1)
    # exit()
    start(fname="C:/Users/fo18103/PycharmProjects/prediction_of_helminths_infection/training_data_generator_and_ml_classifier/src/resolution_10min_days_5_div/training_sets/activity_.data")
    # start(fname="C:/Users/fo18103/PycharmProjects/training_data_generator/src/resolution_10min_days_6/training_sets/cwt_.data")
