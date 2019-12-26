import os

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
import time
from sys import exit

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
        s = s.interpolate(method='linear', limit_direction='both')
        # s = s.interpolate(method='spline', limit_direction='both')
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
    w = pywt.ContinuousWavelet('morl')
    scales = even_list(40)
    sampling_frequency = 1 / 60
    sampling_period = 1 / sampling_frequency
    activity_i = interpolate(activity)
    coef, freqs = pywt.cwt(np.asarray(activity_i), scales, w, sampling_period=sampling_period)
    cwt = [element for tupl in coef for element in tupl]
    indexes = np.asarray(list(range(coef.shape[1])))
    return cwt, coef, freqs, indexes, scales, 1, 'morlet'


def compute_cwt_hd(activity):
    print("compute_cwt...")
    # t, activity = dummy_sin()
    num_steps = len(activity)
    x = np.arange(num_steps)
    y = activity
    y = interpolate(y)

    delta_t = x[1] - x[0]
    scales = np.arange(1, num_steps + 1)/1
    freqs = 1 / (wavelet.Morlet().flambda() * scales)
    wavelet_type = 'morlet'

    # y = [0 if x is np.nan else x for x in y] #todo fix
    coefs, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(y, delta_t, wavelet=wavelet_type, freqs=freqs)

    # print("*********************************************")
    # print(y)
    # print(coefs)
    # print("*******************************************")
    # iwave = wavelet.icwt(coefs, scales, delta_t, wavelet=wavelet_type)
    # plt.plot(iwave)
    # plt.show()
    # plt.plot(activity)
    # plt.show()
    #
    # plt.matshow((coefs.real))
    # plt.show()
    # exit()
    cwt = [element for tupl in coefs.real for element in tupl]
    indexes = np.asarray(list(range(len(coefs.real))))
    return cwt, coefs.real, freqs, indexes, scales, delta_t, wavelet_type


def start(fname='', out_fname=None, id=None):
    print(out_fname, id)
    print("loading dataset...")
    # print(fname)
    data_frame = pd.read_csv(fname, sep=",", header=None)
    # print(data_frame)
    sample_count = data_frame.shape[1]
    hearder = [str(n) for n in range(0, sample_count)]
    hearder[-7] = "class"
    hearder[-6] = "elem_in_row"
    hearder[-5] = "date1"
    hearder[-4] = "date2"
    hearder[-3] = "serial"
    hearder[-2] = "date2"
    hearder[-1] = "serial"
    data_frame.columns = hearder
    data_frame_0 = data_frame
    data_frame = data_frame.loc[:, :'class']
    np.random.seed(0)
    data_frame = data_frame.sample(frac=1).reset_index(drop=True)
    # data_frame = data_frame.fillna(-1)
    data_frame = shuffle(data_frame)
    data_frame['class'] = data_frame['class'].map({True: 1, False: 0})
    process(data_frame, data_frame_0, out_fname, id)


def mean(a):
    return sum(a) / len(a)


def interpolate_(array):
    nans, x = nan_helper(array)
    array[nans] = np.interp(x(nans), x(~nans), array[~nans])
    return array


def to_list_of_nparray(l):
    result = []
    for array in l:
        array = np.asarray(array, dtype=np.float16)
        array = interpolate(array)
        result.append(array)

    return result


def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]


def contains_negative(list):
    for v in list:
        if v is None:
            continue
        if v < 0:
            print("negative number found!", v)
            exit()
            return True
    return False


def purge_file(filename):
    print("purge %s..." % filename)
    try:
        os.remove(filename)
    except FileNotFoundError:
        print("file not found.")

data = []
def process_data_frame(data_frame, data_frame_0, out_fname = None):
    print(out_fname)
    # data_frame = data_frame.fillna(-1)
    X = data_frame[data_frame.columns[0:data_frame.shape[1] - 1]].values
    X_date = data_frame_0[data_frame_0.columns[data_frame_0.shape[1] - 4:data_frame_0.shape[1]]].values
    cwt_list = []
    class0 = []
    class1 = []


    H = []
    with open(out_fname, 'a') as outfile:
        for i, activity in enumerate(X):
            activity = interpolate(activity)
            activity = np.asarray(activity)
            H.append(activity)
    herd_mean = np.average(H, axis=0)

    # herd_mean = np.average(herd_data, axis=0)
    # herd_mean = interpolate(herd_mean)
    cwt_herd, coefs_herd_mean, _, _, _, _, _ = compute_cwt_hd(herd_mean)

    # herd_mean = minmax_scale(herd_mean, feature_range=(0, 1))

    purge_file(out_fname)
    with open(out_fname, 'a') as outfile:
        for i, activity in enumerate(X):
            activity = interpolate(activity)
            activity = np.asarray(activity)

            # activity = minmax_scale(activity, feature_range=(0, 1))
            if 'div' in out_fname:
                activity = np.divide(activity, herd_mean)
            if 'sub' in out_fname:
                activity = np.subtract(activity, herd_mean)
            # activity[activity == np.inf] = np.nan
            # activity = interpolate(activity)

            print("%d/%d ..." % (i, len(X)))
            cwt, coefs, freqs, indexes, scales, delta_t, wavelet_type = compute_cwt(activity)

            data.append({'indexes': indexes, 'coef': coefs, 'freqs': freqs, 'shape': activity.shape[0], 'title': str(i)})
            # create_cwt_graph(coefs, freqs, activity.shape[0], title=str(i))

            cwt_list.append(cwt)
            target = data_frame.at[i, 'class']

            label = 'False'
            if target == 0:
                class0.append(activity)
            if target == 1:
                label = 'True'
                class1.append(activity)

            training_str_flatten = str(coefs.shape).strip('()')+','+str(cwt).strip('[]').replace(' ', '').replace('None', 'NaN')+','+label+','+str(X_date[i].tolist()).replace('\'','').strip('[]').replace(' ','')

            print(" %s.....%s" % (training_str_flatten[0:50], training_str_flatten[-50:]))

            outfile.write(training_str_flatten)
            outfile.write('\n')


    class0_mean = np.average(class0, axis=0)
    # class0_mean[class0_mean == 0] = np.nan
    # class0_mean = interpolate(class0_mean)

    class1_mean = np.average(class1, axis=0)
    # class1_mean[class1_mean == 0] = np.nan
    # class1_mean = interpolate(class1_mean)

    # print("***********************")
    # print(herd_data)
    # print("***********************")
    # print(herd_mean)
    print("***********************")

    # plt.plot(herd_mean)
    # plt.show()
    # plt.plot(class0_mean)
    # plt.show()
    # plt.plot(class1_mean)
    # plt.show()
    #
    #
    # plt.yscale('log')
    # plt.matshow(coefs_herd_mean)
    # plt.show()

    _, coefs_class0_mean, _, _, _, _, _ = compute_cwt_hd(class0_mean)
    # plt.matshow(coefs_class0_mean)
    # plt.show()
    _, coefs_class1_mean, _, _, _, _, _ = compute_cwt_hd(class1_mean)
    # plt.matshow(coefs_class1_mean)
    # plt.show()

    X = pd.DataFrame.from_records(cwt_list)
    # print(X)
    y = data_frame["class"].values.flatten()
    # X = normalize(X)
    # X = preprocessing.MinMaxScaler().fit_transform(X)
    return X.values, y, scales, delta_t, wavelet_type, class0_mean, coefs_class0_mean, class1_mean, coefs_class1_mean, coefs_herd_mean, herd_mean, class0_mean, class1_mean, out_fname


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


def process(data_frame, data_frame_0, out_fname=None, id=None):
    print("process...")
    X, y, scales, delta_t, wavelet_type, class0_mean, coefs_class0_mean, class1_mean, coefs_class1_mean,\
    coefs_herd_mean, herd_mean, class0_mean, class1_mean, out_fname = process_data_frame(data_frame, data_frame_0, out_fname)
    print("train_test_split...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    clf = SVC(kernel='linear')
    # clf = LDA(n_components=2)
    # y_train[0] = 3

    print("fit...")
    clf.fit(X_train, y_train)

    # result = class_feature_importance(X.values, y, clf.coef_[0])
    print("explain_prediction...")
    aux1 = eli5.sklearn.explain_prediction.explain_prediction_linear_classifier(clf, X[0], top=X.shape[1])
    aux1 = eli5.format_as_dataframe(aux1)
    print("********************************")
    print(aux1)
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

    #value0 = class0['value'].values
    weight0 = class0['weight'].values

    #value1 = class1['value'].values
    weight1 = class1['weight'].values

    #value0 = pad(value0, data[0]['coef'].shape[0] * data[0]['coef'].shape[1])
    weight0 = pad(weight0, data[0]['coef'].shape[0] * data[0]['coef'].shape[1])
    #value1 = pad(value1, data[0]['coef'].shape[0] * data[0]['coef'].shape[1])
    weight1 = pad(weight1, data[0]['coef'].shape[0] * data[0]['coef'].shape[1])

    fig, axs = plt.subplots(5, 2)
    fig.set_size_inches(25, 25)
    outfile = 'SVC_%s.png' % out_fname.split('.')[0]
    axs[0, 0].set_title(outfile, fontsize=25, loc="left", pad=30)

    ymin = min([min(class0_mean), min(class1_mean), min(herd_mean)])
    ymax = max([max(class0_mean), max(class1_mean), max(herd_mean)])

    axs[0, 0].pcolor(data[0]['indexes'], data[0]['freqs'], coefs_class0_mean)
    axs[0, 0].set_yscale('log')
    axs[0, 0].set_title('class0 cwt input')
    # iwave = wavelet.icwt(coefs_class0_mean, scales, delta_t, wavelet=wavelet_type)
    # iwave = np.real(iwave)
    axs[0, 1].plot(class0_mean)
    axs[0, 1].set_ylim([ymin, ymax])
    axs[0, 1].set_title('class0 time input')

    axs[1, 0].pcolor(data[0]['indexes'], data[0]['freqs'], coefs_class1_mean)
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_title('class1 cwt input')
    # iwave = wavelet.icwt(coefs_class1_mean, scales, delta_t, wavelet=wavelet_type)
    # iwave = np.real(iwave)
    axs[1, 1].plot(class1_mean)
    axs[1, 1].set_ylim([ymin, ymax])
    axs[1, 1].set_title('class1 time input')


    c0 = np.reshape(normalized(weight0), data[0]['coef'].shape)
    iwave0 = wavelet.icwt(c0, scales, delta_t, wavelet=wavelet_type)
    iwave0 = np.real(iwave0)

    c1 = np.reshape(normalized(weight1), data[0]['coef'].shape)
    iwave1 = wavelet.icwt(c1, scales, delta_t, wavelet=wavelet_type)
    iwave1 = np.real(iwave1)

    ymin2 = min([min(iwave0), min(iwave1)])
    ymax2 = max([max(iwave0), max(iwave1)])

    axs[2, 0].pcolor(data[0]['indexes'], data[0]['freqs'], c0)
    axs[2, 0].set_yscale('log')
    axs[2, 0].set_title('class0 cwt weight')
    axs[2, 1].plot(iwave0)
    axs[2, 1].set_ylim([ymin2, ymax2])
    axs[2, 1].set_title('class0 cwt weight inverse')

    axs[3, 0].pcolor(data[0]['indexes'], data[0]['freqs'], c1)
    axs[3, 0].set_yscale('log')
    axs[3, 0].set_title('class1 cwt weight')
    axs[3, 1].plot(iwave1)
    axs[3, 1].set_ylim([ymin2, ymax2])
    axs[3, 1].set_title('class1 cwt weight inverse')


    axs[4, 0].pcolor(data[0]['indexes'], data[0]['freqs'], coefs_herd_mean)
    # Set yscale, ylim and labels
    axs[4, 0].set_yscale('log')
    axs[4, 0].set_title('mean cwt input')
    # iwave_m = wavelet.icwt(coefs_herd_mean, scales, delta_t, wavelet=wavelet_type)
    # iwave_m = np.real(iwave_m)
    axs[4, 1].set_ylim([ymin, ymax])
    axs[4, 1].plot(herd_mean)
    axs[4, 1].set_title('mean time input')

    # for ax in axs.flat:
    #     ax.label_outer()

    fig.show()
    # outfile = 'SVC_%s.png' % str(time.time()).split('.')[0]
    fig.savefig(str(id)+'_'+outfile, dpi=100)
    # exit()



if __name__ == '__main__':
    # dir = 'E:/Users/fo18103/PycharmProjects/prediction_of_helminths_infection/training_data_generator_and_ml_classifier/src/delmas_70101200027_resolution_10min_days_6/'
    # with open("%s/herd_activity.json" % dir, "r") as read_file:
    #     herd_data = json.load(read_file)

    #start(fname="%s/training_sets/activity_.data" % dir, out_fname='2_cwt_.data', id=2)
    #start(fname="%s/training_sets/activity_.data" % dir, out_fname='2_cwt_sub.data', id=2)
    #start(fname="%s/training_sets/activity_.data" % dir, out_fname='2_cwt_div.data', id=2)

    dir = 'E:/Users/fo18103/PycharmProjects/prediction_of_helminths_infection/training_data_generator_and_ml_classifier/src/delmas_70101200027_resolution_10min_days_12/'
    # with open("%s/herd_activity.json" % dir, "r") as read_file:
    #     herd_data = json.load(read_file)

    # start(fname="%s/training_sets/activity_.data" % dir, out_fname='4_cwt_.data', id=4)
    # start(fname="%s/training_sets/activity_.data" % dir, out_fname='4_cwt_sub.data', id=4)
    start(fname="%s/training_sets/activity_.data" % dir, out_fname='5_cwt_div.data', id=5)
