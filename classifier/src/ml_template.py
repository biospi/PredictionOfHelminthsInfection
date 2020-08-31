import os

import pandas as pd
import numpy as np
import pywt
import pycwt as wavelet
from sklearn import preprocessing
from scipy.interpolate import UnivariateSpline
from sklearn.preprocessing import normalize, minmax_scale
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import pathlib
from sklearn import datasets
from mlxtend.plotting import plot_decision_regions
import scikitplot as skplt
from sklearn.cross_decomposition import PLSRegression


def even_list(n):
    result = [1]
    for num in range(2, n * 2 + 1, 2):
        result.append(num)
    del result[-1]
    return np.asarray(result, dtype=np.int32)


def compute_cwt(activity, hd=True):
    if hd:
        return compute_cwt_hd(activity)
    else:
        return compute_cwt_sd(activity)


def compute_cwt_sd(activity):
    w = pywt.ContinuousWavelet('morl')
    scales = even_list(20)
    sampling_frequency = 1 / 60
    sampling_period = 1 / sampling_frequency
    activity_i = interpolate(activity)
    coef, freqs = pywt.cwt(np.asarray(activity_i), scales, w, sampling_period=sampling_period)
    cwt = [element for tupl in coef for element in tupl]
    # indexes = np.asarray(list(range(coef.shape[1])))
    indexes = []
    return cwt, coef, freqs, indexes, scales, 1, 'morlet'


def compute_cwt_hd(activity):
    print("compute_cwt...")
    num_steps = len(activity)
    x = np.arange(num_steps)
    y = activity
    y = interpolate(y)

    delta_t = (x[1] - x[0]) * 1
    scales = np.arange(1, num_steps + 1) / 1
    freqs = 1 / (wavelet.Morlet().flambda() * scales)
    wavelet_type = 'morlet'
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
    # indexes = np.asarray(list(range(len(coefs.real))))
    indexes = []
    return cwt, coefs.real, freqs, indexes, scales, delta_t, wavelet_type


def interpolate(input_activity):
    try:
        i = np.array(input_activity, dtype=np.float)
        s = pd.Series(i)
        s = s.interpolate(method='linear', limit_direction='both')
        return s.tolist()
    except ValueError as e:
        print(e)
        return input_activity


def interpolate_time(a, new_length):
    old_indices = np.arange(0, len(a))
    new_indices = np.linspace(0, len(a) - 1, new_length)
    spl = UnivariateSpline(old_indices, a, k=3, s=0)
    new_array = spl(new_indices)
    # new_array[0] = 0
    return new_array


def get_cwt_data_frame(data_frame):
    X = data_frame[data_frame.columns[0:data_frame.shape[1] - 1]].values

    X_cwt = pd.DataFrame()
    cpt = 0
    class0_freq_domain = []
    class1_freq_domain = []
    class0_time_domain = []
    class1_time_domain = []
    for activity, (i, row) in zip(X, data_frame.iterrows()):
        activity = interpolate(activity)
        activity = np.asarray(activity)
        print(len(activity), "%d/%d ..." % (cpt, len(X)))
        cwt, coefs, freqs, indexes, scales, delta_t, wavelet_type = compute_cwt(activity)

        if cpt == 0:
            X_cwt = pd.DataFrame(columns=[str(x) for x in range(len(cwt))], dtype=np.float16)

        X_cwt.loc[cpt] = cwt
        cpt += 1
        target = data_frame.at[i, 'label']

        if target == 0:
            class0_freq_domain.append(cwt)
            class0_time_domain.append(activity)
        if target == 1:
            class1_freq_domain.append(cwt)
            class1_time_domain.append(activity)

    class0_mean = np.average(class0_time_domain, axis=0)
    _, coefs_class0_mean, _, _, _, _, _ = compute_cwt(class0_mean)
    class1_mean = np.average(class1_time_domain, axis=0)
    _, coefs_class1_mean, _, _, _, _, _ = compute_cwt(class1_mean)

    fig, axs = plt.subplots(2, 2, facecolor='white')
    time_axis = range(0, len(class0_mean))

    im = axs[0, 0].pcolormesh(time_axis, freqs, coefs_class0_mean, cmap='viridis')
    fig.colorbar(im, ax=axs[0, 0])
    axs[0, 0].set_title('Magnitude')
    # fig.colorbar(im, ax=ax)
    axs[0, 0].set_title('Continuous Wavelet Transform')
    axs[0, 0].set_yscale('log')
    axs[0, 0].set(xlabel="$Time (days)$", ylabel="$Frequency$")

    axs[0, 1].bar(time_axis, class0_mean)


    im = axs[1, 0].pcolormesh(time_axis, freqs, coefs_class1_mean, cmap='viridis')
    fig.colorbar(im, ax=axs[1, 0])
    axs[1, 0].set_title('Magnitude')
    # fig.colorbar(im, ax=ax)
    axs[1, 0].set_title('Continuous Wavelet Transform')
    axs[1, 0].set_yscale('log')
    axs[1, 0].set(xlabel="$Time (days)$", ylabel="$Frequency$")

    axs[1, 1].bar(time_axis, class1_mean)



    # plt.bar(range(0, len(class0_mean)), class0_mean)
    # plt.show()
    # plt.pcolormesh(coefs_class0_mean, cmap='viridis')
    # plt.show()
    #
    # plt.bar(range(0, len(class1_mean)), class1_mean)
    # plt.show()
    # plt.pcolormesh(coefs_class1_mean, cmap='viridis')
    # plt.show()

    fig.show()

    y = data_frame["label"].values.flatten().astype(int)
    X_cwt['label'] = y

    print("*********************************")
    print(data_frame)
    print(X_cwt)
    print("*********************************")
    return X_cwt, (scales, delta_t, wavelet_type, class0_mean, coefs_class0_mean, class1_mean, coefs_class1_mean)


def process_data_frame_(data_frame, y_col='label'):
    #standarization step before ml
    data_frame = data_frame.fillna(-1)
    X = data_frame[data_frame.columns[2:data_frame.shape[1] - 1]].values
    X = normalize(X)
    X = preprocessing.MinMaxScaler().fit_transform(X)
    y = data_frame[y_col].values.flatten()
    y = y.astype(int)
    return X, y


def reduce_pls(output_dim, X, y):
    print("reduce pls...")
    clf = PLSRegression(n_components=output_dim)
    X_pls = clf.fit_transform(X, y)[0]
    return X_pls, y


def reduce_lda(output_dim, X, y):
    # lda implementation require 3 input class for 2d output and 4 input class for 3d output
    # if output_dim not in [1, 2, 3]:
    #     raise ValueError("available dimension for features reduction are 1, 2 and 3.")
    if output_dim == 3:
        X = np.vstack((X, np.array([np.zeros(X.shape[1]), np.ones(X.shape[1])])))
        y = np.append(y, (3, 4))
    if output_dim == 2:
        X = np.vstack((X, np.array([np.zeros(X.shape[1])])))
        y = np.append(y, 3)
    clf = LDA(n_components=output_dim)
    X_lda = clf.fit_transform(X, y)
    if output_dim != 1:
        X_lda = X_lda[0:-(output_dim - 1)]
        y_lda = y[0:-(output_dim - 1)]

    return X_lda, y_lda


def plot_2D_decision_boundaries(X, y, X_test, title, clf, save=True):
    fig = plt.figure(figsize=(8, 7), dpi=100)
    plt.subplots_adjust(top=0.80)
    scatter_kwargs = {'s': 120, 'edgecolor': None, 'alpha': 0.7}
    contourf_kwargs = {'alpha': 0.2}
    scatter_highlight_kwargs = {'s': 120, 'label': 'Test data', 'alpha': 0.7}
    plot_decision_regions(X, y, clf=clf, legend=2,
                          X_highlight=X_test,
                          scatter_kwargs=scatter_kwargs,
                          contourf_kwargs=contourf_kwargs,
                          scatter_highlight_kwargs=scatter_highlight_kwargs)
    plt.title(title)
    if save:
        path = "/decision_boundaries_graphs/"
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        filename = "scatter.png"
        final_path = '%s/%s' % (path, filename)
        print(final_path)
        try:
            plt.savefig(final_path)
        except FileNotFoundError as e:
            print(e)
            exit()

        plt.close()
        fig.show()
        return final_path
    else:
        plt.show()


# def plot_2D_decision_boundaries(X_lda, y_lda, X_test, y_test, title, clf, folder=None, sub_dir_name=None, n_bin=8, save=True):
#     print('graph...')
#     # plt.subplots_adjust(top=0.75)
#     # fig = plt.figure(figsize=(7, 6), dpi=100)
#     fig, ax = plt.subplots(figsize=(7., 4.8))
#     # plt.subplots_adjust(top=0.75)
#     min = abs(X_lda.min()) + 1
#     max = abs(X_lda.max()) + 1
#     print(X_lda.shape)
#     print(min, max)
#     if np.max([min, max]) > 100:
#         return
#     xx, yy = np.mgrid[-min:max:.01, -min:max:.01]
#     grid = np.c_[xx.ravel(), yy.ravel()]
#     probs = clf.predict_proba(grid)[:, 1].reshape(xx.shape)
#     offset_r = 0
#     offset_g = 0
#     offset_b = 0
#     colors = [((77+offset_r)/255, (157+offset_g)/255, (210+offset_b)/255),
#               (1, 1, 1),
#               ((255+offset_r)/255, (177+offset_g)/255, (106+offset_b)/255)]
#     cm = LinearSegmentedColormap.from_list('name', colors, N=n_bin)
#
#     for _ in range(0, 1):
#         contour = ax.contourf(xx, yy, probs, n_bin, cmap=cm, antialiased=False, vmin=0, vmax=1, alpha=0.3, linewidth=0,
#                               linestyles='dashed', zorder=-1)
#         ax.contour(contour, cmap=cm, linewidth=1, linestyles='dashed', zorder=-1, alpha=1)
#
#     ax_c = fig.colorbar(contour)
#
#     ax_c.set_alpha(1)
#     ax_c.draw_all()
#
#     ax_c.set_label("$P(y = 1)$")
#
#     X_lda_0 = X_lda[y_lda == 0]
#     X_lda_1 = X_lda[y_lda == 1]
#
#     X_lda_0_t = X_test[y_test == 0]
#     X_lda_1_t = X_test[y_test == 1]
#     marker_size = 150
#     ax.scatter(X_lda_0[:, 0], X_lda_0[:, 1], c=(39/255, 111/255, 158/255), s=marker_size, vmin=-.2, vmax=1.2,
#                edgecolor=(49/255, 121/255, 168/255), linewidth=0, marker='s', alpha=0.7, label='Class0 (Healthy)'
#                , zorder=1)
#
#     ax.scatter(X_lda_1[:, 0], X_lda_1[:, 1], c=(251/255, 119/255, 0/255), s=marker_size, vmin=-.2, vmax=1.2,
#                edgecolor=(255/255, 129/255, 10/255), linewidth=0, marker='^', alpha=0.7, label='Class1 (Unhealthy)'
#                , zorder=1)
#
#     ax.scatter(X_lda_0_t[:, 0], X_lda_0_t[:, 1], s=marker_size-10, vmin=-.2, vmax=1.2,
#                edgecolor="black", facecolors='none', label='Test data', zorder=1)
#
#     ax.scatter(X_lda_1_t[:, 0], X_lda_1_t[:, 1], s=marker_size-10, vmin=-.2, vmax=1.2,
#                edgecolor="black", facecolors='none', zorder=1)
#
#     ax.set(xlabel="$X_1$", ylabel="$X_2$")
#
#     ax.contour(xx, yy, probs, levels=[.5], cmap="Reds", vmin=0, vmax=.6, linewidth=0.1)
#
#     for spine in ax.spines.values():
#         spine.set_edgecolor('white')
#
#     handles, labels = ax.get_legend_handles_labels()
#     db_line = Line2D([0], [0], color=(183/255, 37/255, 42/255), label='Decision boundary')
#     handles.append(db_line)
#
#     plt.legend(loc=4, fancybox=True, framealpha=0.4, handles=handles)
#     plt.title(title)
#     ttl = ax.title
#     ttl.set_position([.57, 0.97])
#
#     if save:
#         path = "%s/%s/decision_boundaries_graphs/df%d/" % (folder, sub_dir_name, 0)
#         pathlib.Path(path).mkdir(parents=True, exist_ok=True)
#         filename = "iter_%d.png" % (0)
#         final_path = '%s/%s' % (path, filename)
#         print(final_path)
#         try:
#             plt.savefig(final_path, bbox_inches='tight')
#         except FileNotFoundError as e:
#             print(e)
#             exit()
#
#         plt.close()
#         # fig.show()
#         plt.close()
#         fig.clear()
#     else:
#         fig.show()
#

def purge_file(filename):
    print("purge %s..." % filename)
    try:
        os.remove(filename)
    except Exception:
        print("file not found.")


def generate_fake_data_to_import():
    fname = 'dummy_data.csv'
    purge_file(fname)
    X, y = datasets.make_blobs(n_samples=50, centers=2, n_features=100, center_box=(0, 10))

    for i in range(10):
        print(i)
        X = np.vstack((X, np.array([np.zeros(X.shape[1])])))
        y = np.append(y, np.random.randint(2, size=1)[0])

        # X = np.vstack((X, np.array([np.ones(X.shape[1])*-1])))
        # y = np.append(y, np.random.randint(2, size=1)[0])

    df = pd.DataFrame(X)
    df, hearder = rename_df_cols(df)
    df['label'] = y
    print(df)
    print('labels')
    print(df['label'].value_counts())
    df.to_csv(fname, sep=',', index=False)
    return fname, hearder


def rename_df_cols(df):
    sample_count = df.shape[1]
    hearder = ["feature%d" % n for n in range(0, sample_count)]
    df.columns = hearder
    return df, hearder


def print_proba_report(X_test, y_probas):
    pd.options.display.float_format = '{:.10f}'.format
    df_y_probas = pd.DataFrame(y_probas)
    df_y_probas.columns = ['sample proba class0', 'sample proba class1']
    print(df_y_probas)


def save_roc_curve(y_test, y_probas):
    title = 'ROC Curve'
    skplt.metrics.plot_roc(y_test, y_probas, title=title, title_fontsize='medium')
    path = "/roc_curve/"
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    final_path = '%s/%s' % (path, 'roc.png')
    print(final_path)
    plt.savefig(final_path)
    plt.show()
    plt.close()


def f_importances(coef, names):
    imp = coef
    imp, names = zip(*sorted(zip(imp, names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()


if __name__ == "__main__":
    print("start...")

    activity = list([np.nan]*10)
    cwt, coefs, freqs, indexes, scales, delta_t, wavelet_type = compute_cwt(activity)


    fname, features_names = generate_fake_data_to_import()
    df = pd.read_csv(fname, sep=",")

    df_cwt, _ = get_cwt_data_frame(df)
    X, y = process_data_frame_(df_cwt)
    ENABLE_LDA_DR = True

    if ENABLE_LDA_DR:
        features_names = ['lda componant0', 'lda componant1']
        X_lda, y = reduce_lda(2, X, y)

    print("fitting...")
    clf = SVC(kernel='linear', probability=True)
    # clf = LogisticRegression(C=1e10)

    X_train, X_test, y_train, y_test = train_test_split(X_lda if ENABLE_LDA_DR else X, y,
                                                        test_size=0.4, random_state=0, stratify=y)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(acc)
    print(classification_report(y_test, y_pred))
    y_proba = clf.predict_proba(X_test)
    print_proba_report(X_test, y_proba)
    # save_roc_curve(y_test, y_proba)

    if ENABLE_LDA_DR:
        plot_2D_decision_boundaries(X_lda, y, X_test, 'title', clf, save=True)

    # if isinstance(clf, LogisticRegression):
    #     f_importances(clf.coef_[0], features_names)
    # if isinstance(clf, SVC):
    #     f_importances(clf.coef_, features_names)