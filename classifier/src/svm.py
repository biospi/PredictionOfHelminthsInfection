from datetime import datetime
import math
import pathlib
import shutil
import warnings
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from numpy.random import random
import matplotlib.font_manager
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import scikitplot as skplt
from matplotlib.lines import Line2D
from mlxtend.plotting import plot_decision_regions
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold, RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.utils import shuffle
# import plotly.plotly as py
# import plotly.graph_objs as go
# import plotly.offline as py
# import plotly.tools as tls
# from IPython.display import display
# import plotly
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sys import exit
from matplotlib.lines import Line2D
from scipy import interp
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import auc
import scipy.stats
from sklearn.cross_decomposition import PLSRegression

from sklearn import metrics
import os

os.environ['R_HOME'] = 'C:\Program Files\R\R-3.6.1'  # path to your R installation
os.environ[
    'R_USER'] = 'C:\\Users\\fo18103\\AppData\\Local\Continuum\\anaconda3\Lib\site-packages\\rpy2'  # path depends on where you installed Python. Mine is the Anaconda distribution

import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr, isinstalled

utils = importr('utils')
R = robjects.r

# need to be installed from Rstudio or other package installer
print('e1071', isinstalled('e1071'))
print('rgl', isinstalled('rgl'))
print('misc3d', isinstalled('misc3d'))
print('plot3D', isinstalled('plot3D'))
print('plot3Drgl', isinstalled('plot3Drgl'))

if not isinstalled('e1071'):
    utils.install_packages('e1071')
if not isinstalled('rgl'):
    utils.install_packages('rgl')
if not isinstalled('misc3d'):
    utils.install_packages('misc3d')
if not isinstalled('plot3D'):
    utils.install_packages('plot3D')
if not isinstalled('plot3Drgl'):
    utils.install_packages('plot3Drgl')

e1071 = importr('e1071')
rgl = importr('rgl')
misc3d = importr('misc3d')
plot3D = importr('plot3D')
plot3Drgl = importr('plot3Drgl')

print(rpy2.__version__)

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 10)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

np.random.seed(0)
MAIN_DIR = "E:/Users/fo18103/PycharmProjects/prediction_of_helminths_infection/training_data_generator_and_ml_classifier/src/20min/"
META_DATA_LENGTH = 19


def even_list(n):
    result = [1]
    for num in range(2, n * 2 + 1, 2):
        result.append(num)
    del result[-1]
    return np.asarray(result, dtype=np.int32)


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


def compute_cwt(activity):
    name = 'gaus8'
    w = pywt.ContinuousWavelet(name)
    scales = even_list(40)
    sampling_frequency = 1 / 60
    sampling_period = 1 / sampling_frequency
    activity_i = interpolate(activity)
    coef, freqs = pywt.cwt(np.asarray(activity_i), scales, w, sampling_period=sampling_period)
    cwt = coef.flatten().tolist()
    indexes = list(range(len(cwt)))
    indexes.reverse()
    return cwt, coef, freqs, indexes


def percentage(whole, percent):
    result = int((whole * percent) / 100.0)
    print("%d percent of %d is %d." % (percent, whole, result))
    return result


def create_cwt_graph(coef, lenght, title=None):
    time = [x for x in range(0, lenght)]
    fig = plt.figure()
    freq = [x for x in range(0, 40)]
    freq = np.array(freq)
    plt.pcolormesh(time, freq, coef)
    fig.suptitle(title, x=0.5, y=.95, horizontalalignment='center', verticalalignment='top', fontsize=10)
    fig.show()


def process_data_frame(data_frame, y_col='label'):
    data_frame = data_frame.fillna(-1)
    cwt_shape = data_frame[data_frame.columns[0:2]].values
    X = data_frame[data_frame.columns[2:data_frame.shape[1] - META_DATA_LENGTH]].values
    print(X)
    X = normalize(X, norm='max')
    X = preprocessing.MinMaxScaler().fit_transform(X)
    y = data_frame[y_col].values.flatten()
    y = y.astype(int)
    return X, y


def process_and_split_data_frame(data_frame):
    data_frame = data_frame.fillna(-1)
    X = data_frame[data_frame.columns[0:data_frame.shape[1] - 1]].values
    X = normalize(X)
    X = preprocessing.MinMaxScaler().fit_transform(X)
    # print(X.shape, X)
    # print(DataFrame.from_records(X))
    y = data_frame["class"].values.flatten()
    train_x, test_x, train_y, test_y = train_test_split(X, y, train_size=0.7, shuffle=True)
    return X, y, train_x, test_x, train_y, test_y


def get_prec_recall_fscore_support(test_y, pred_y):
    precision_recall_fscore_support_result = precision_recall_fscore_support(test_y, pred_y, average=None,
                                                                             labels=[0, 1])
    precision_false = precision_recall_fscore_support_result[0][0]
    precision_true = precision_recall_fscore_support_result[0][1]
    recall_false = precision_recall_fscore_support_result[1][0]
    recall_true = precision_recall_fscore_support_result[1][1]
    fscore_false = precision_recall_fscore_support_result[2][0]
    fscore_true = precision_recall_fscore_support_result[2][1]
    support_false = precision_recall_fscore_support_result[3][0]
    support_true = precision_recall_fscore_support_result[3][1]
    return precision_false, precision_true, recall_false, recall_true, fscore_false, fscore_true, support_false, support_true


def plot_2D_decision_boundaries(X_lda, y_lda, X_test, y_test, title, clf, filename="", days=None, resolution=None, n_bin=8
                                , style2=True, i=0):
    print('graph...')
    # plt.subplots_adjust(top=0.75)
    # fig = plt.figure(figsize=(7, 6), dpi=100)
    fig, ax = plt.subplots(figsize=(7., 4.8))
    # plt.subplots_adjust(top=0.75)
    min = abs(X_lda.min()) + 1
    max = abs(X_lda.max()) + 1
    step = 0.1
    print(min, max, step)
    xx, yy = np.mgrid[-min:max:step, -min:max:step]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = clf.predict_proba(grid)[:, 1].reshape(xx.shape)
    offset_r = 0
    offset_g = 0
    offset_b = 0
    colors = [((77+offset_r)/255, (157+offset_g)/255, (210+offset_b)/255),
              (1, 1, 1),
              ((255+offset_r)/255, (177+offset_g)/255, (106+offset_b)/255)]
    if style2:
        colors = [((77 + offset_r) / 255, (157 + offset_g) / 255, (210 + offset_b) / 255),
                  (1, 1, 1),
                  ((255 + offset_r) / 255, (177 + offset_g) / 255, (106 + offset_b) / 255)]

    cm = LinearSegmentedColormap.from_list('name', colors, N=n_bin)

    for _ in range(0, 1):
        contour = ax.contourf(xx, yy, probs, n_bin, cmap=cm, antialiased=False, vmin=0, vmax=1, alpha=0.3, linewidth=0,
                              linestyles='dashed', zorder=-1)
        ax.contour(contour, cmap=cm, linewidth=1, linestyles='dashed', zorder=-1, alpha=1)

    ax_c = fig.colorbar(contour)
    # tick_locator = ticker.MaxNLocator(nbins=4)
    # ax_c.locator = tick_locator
    # ax_c.update_ticks()

    ax_c.set_alpha(1)
    ax_c.draw_all()

    ax_c.set_label("$P(y = 1)$")
    # ax_c.set_ticks([0, .25, 0.5, 0.75, 1])
    # ax_c.ax.set_yticklabels(['0', '0.5', '1', '0.5', '0'])

    X_lda_0 = X_lda[y_lda == 0]
    X_lda_1 = X_lda[y_lda == 1]

    X_lda_0_t = X_test[y_test == 0]
    X_lda_1_t = X_test[y_test == 1]
    marker_size = 150
    if style2:
        ax.scatter(X_lda_0[:, 0], X_lda_0[:, 1], c=(39/255, 111/255, 158/255), s=marker_size, vmin=-.2, vmax=1.2,
                   edgecolor=(49/255, 121/255, 168/255), linewidth=0, marker='s', alpha=0.7, label='Class0 (Healthy)'
                   , zorder=1)

        ax.scatter(X_lda_1[:, 0], X_lda_1[:, 1], c=(251/255, 119/255, 0/255), s=marker_size, vmin=-.2, vmax=1.2,
                   edgecolor=(255/255, 129/255, 10/255), linewidth=0, marker='^', alpha=0.7, label='Class1 (Unhealthy)'
                   , zorder=1)

        ax.scatter(X_lda_0_t[:, 0], X_lda_0_t[:, 1], c=(43/255, 75/255, 98/255), marker='s', s=marker_size, vmin=-.2, vmax=1.2,
                   edgecolor="black", label='Test data Class0', zorder=1, alpha=0.6)

        ax.scatter(X_lda_1_t[:, 0], X_lda_1_t[:, 1], c=(182/255, 83/255, 10/255), marker='^', s=marker_size, vmin=-.2, vmax=1.2,
                   edgecolor="black", zorder=1, label='Test data Class1', alpha=0.6)
    else:
        ax.scatter(X_lda_0[:, 0], X_lda_0[:, 1], c=(39 / 255, 111 / 255, 158 / 255), s=marker_size, vmin=-.2, vmax=1.2,
                   edgecolor=(49 / 255, 121 / 255, 168 / 255), linewidth=0, marker='s', alpha=0.7,
                   label='Class0 (Healthy)'
                   , zorder=1)

        ax.scatter(X_lda_1[:, 0], X_lda_1[:, 1], c=(251 / 255, 119 / 255, 0 / 255), s=marker_size, vmin=-.2, vmax=1.2,
                   edgecolor=(255 / 255, 129 / 255, 10 / 255), linewidth=0, marker='^', alpha=0.7,
                   label='Class1 (Unhealthy)'
                   , zorder=1)

        ax.scatter(X_lda_0_t[:, 0], X_lda_0_t[:, 1], s=marker_size - 10, vmin=-.2, vmax=1.2,
                   edgecolor="black", facecolors='none', label='Test data', zorder=1)

        ax.scatter(X_lda_1_t[:, 0], X_lda_1_t[:, 1], s=marker_size - 10, vmin=-.2, vmax=1.2,
                   edgecolor="black", facecolors='none', zorder=1)

    ax.set(xlabel="$X_1$", ylabel="$X_2$")

    ax.contour(xx, yy, probs, levels=[.5], cmap="Reds", vmin=0, vmax=.6, linewidth=0.1)

    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    handles, labels = ax.get_legend_handles_labels()
    db_line = Line2D([0], [0], color=(183/255, 37/255, 42/255), label='Decision boundary')
    handles.append(db_line)

    plt.legend(loc=2, fancybox=True, framealpha=0.4, handles=handles)
    plt.title(title)
    ttl = ax.title
    ttl.set_position([.57, 0.97])
    # plt.tight_layout()

    path = filename + '\\' + str(resolution) + '\\'
    path_file = path + "%d_%d_p.png" % (days, i)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    plt.savefig(path_file, bbox_inches='tight')
    print(path_file)
    plt.show()
    plt.close()


def plot_2D_decision_boundaries_(X, y, X_test, title, clf, i=0, filename="", days=None, resolution=None):
    plt.subplots_adjust(top=0.75)
    scatter_kwargs = {'s': 120, 'edgecolor': None, 'alpha': 0.7}
    contourf_kwargs = {'alpha': 0.2}
    scatter_highlight_kwargs = {'s': 120, 'label': 'Test data', 'alpha': 0.7}
    plot_decision_regions(X, y, clf=clf, legend=2,
                          X_highlight=X_test,
                          scatter_kwargs=scatter_kwargs,
                          contourf_kwargs=contourf_kwargs,
                          scatter_highlight_kwargs=scatter_highlight_kwargs)
    plt.title(title)
    path = filename + '\\' + str(resolution) + '\\'
    path_file = path + "%d.png" % days
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    plt.savefig(path_file)
    plt.show()
    plt.close()


def plot_3D_decision_boundaries(train_x, train_y, test_x, test_y, title, clf, i=0, filename=""):
    R('r3dDefaults$windowRect <- c(0,50, 1000, 1000) ')
    R('open3d()')
    plot3ddb = R('''
    plot3ddb<-function(nnew, group, dat, kernel_, gamma_, coef_, cost_, tolerance_, probability_, test_x_, fitted_, title_, filepath){
            set.seed(12345)
            fit = svm(group ~ ., data=dat, kernel=kernel_, gamma=gamma_, coef0=coef_, cost=cost_, tolerance=tolerance_, fitted= fitted_, probability= probability_)
            x = dat[,-1]$X1
            y = dat[,-1]$X2
            z = dat[,-1]$X3
            x_test = test_x_[,-1]$X1
            y_test = test_x_[,-1]$X2
            z_test = test_x_[,-1]$X3
            i <- 1
            g = dat$group
            x_1 <- list()
            y_1 <- list()
            z_1 <- list()
            x_2 <- list()
            y_2 <- list()
            z_2 <- list()
            for(var in g){
                if(!(x[i] %in% x_test) & !(y[i] %in% y_test)){
                    if (var == 1){
                        x_1 <- append(x_1, x[i])
                        y_1 <- append(y_1, y[i])
                        z_1 <- append(z_1, z[i])
                    }else{
                        x_2 <- append(x_2, x[i])
                        y_2 <- append(y_2, y[i])
                        z_2 <- append(z_2, z[i])
                      }
                }
              i <- i + 1
            }

            x_1 = as.numeric(x_1)
            y_1 = as.numeric(y_1)
            z_1 = as.numeric(z_1)

            x_2 = as.numeric(x_2)
            y_2 = as.numeric(y_2)
            z_2 = as.numeric(z_2)


            j <- 1
            g_test = test_x_$class
            x_1_test <- list()
            y_1_test <- list()
            z_1_test <- list()
            x_2_test <- list()
            y_2_test <- list()
            z_2_test <- list()
            for(var_test in g_test){
              if (var_test == 1){
                x_1_test <- append(x_1_test, x_test[j])
                y_1_test <- append(y_1_test, y_test[j])
                z_1_test <- append(z_1_test, z_test[j])
              }else{
                x_2_test <- append(x_2_test, x_test[j])
                y_2_test <- append(y_2_test, y_test[j])
                z_2_test <- append(z_2_test, z_test[j])
              }

              j <- j + 1
            }

            x_1_test = as.numeric(x_1_test)
            y_1_test = as.numeric(y_1_test)
            z_1_test = as.numeric(z_1_test)

            x_2_test = as.numeric(x_2_test)
            y_2_test = as.numeric(y_2_test)
            z_2_test = as.numeric(z_2_test)

            pch3d(x_2, y_2, z_2, pch = 24, bg = "#f19c51", color = "#f19c51", radius=0.4, alpha = 0.8)
            pch3d(x_1, y_1, z_1, pch = 22, bg = "#6297bb", color = '#6297bb', radius=0.4, alpha = 1)

            pch3d(x_1_test, y_1_test, z_1_test, pch = 22, bg = "#6297bb", color = 'red', radius=0.4, alpha = 0.8)
            pch3d(x_2_test, y_2_test, z_2_test, pch = 24, bg = "#f19c51", color = "red", radius=0.4, alpha = 1)

            newdat.list = lapply(test_x_[,-1], function(x) seq(min(x), max(x), len=nnew))
            newdat      = expand.grid(newdat.list)
            newdat.pred = predict(fit, newdata=newdat, decision.values=T)
            newdat.dv   = attr(newdat.pred, 'decision.values')
            newdat.dv   = array(newdat.dv, dim=rep(nnew, 3))
            grid3d(c("x", "y+", "z"))
            view3d(userMatrix = structure(c(0.850334823131561, -0.102673642337322, 
                                    0.516127586364746, 0, 0.526208400726318, 0.17674557864666, 
                                    -0.831783592700958, 0, -0.00582099659368396, 0.978886127471924, 
                                    0.20432074368, 0, 0, 0, 0, 1)))

            decorate3d(box=F, axes = T, xlab = '', ylab='', zlab='', aspect = FALSE, expand = 1.03)
            light3d(diffuse = "gray", specular = "gray")
            contour3d(newdat.dv, level=0, x=newdat.list$X1, y=newdat.list$X2, z=newdat.list$X3, add=T, alpha=0.8, plot=T, smooth = 200, color='#28b99d', color2='#28b99d')
            bgplot3d({
                      plot.new()
                      title(main = title_, line = -8, outer=F)
                      #mtext(side = 1, 'This is a subtitle', line = 4)
                      legend("bottomleft", inset=.1,
                               pt.cex = 2,
                               cex = 1, 
                               bty = "n", 
                               legend = c("Decision boundary", "Class 0", "Class 1", "Test data"), 
                               col = c("#28b99d", "#6297bb", "#f19c51", "red"), 
                               pch = c(15, 15,17, 1))
            })
            rgl.snapshot(filepath, fmt="png", top=TRUE)
    }''')

    nnew = test_x.shape[0]
    gamma = clf.best_params_['gamma']
    coef0 = clf.estimator.coef0
    cost = clf.best_params_['C']
    tolerance = clf.estimator.tol
    probability_ = clf.estimator.probability

    df = pd.DataFrame(train_x)
    df.insert(loc=0, column='group', value=train_y + 1)
    df.columns = ['group', 'X1', 'X2', 'X3']
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    r_dataframe = pandas2ri.py2ri(df)

    df_test = pd.DataFrame(test_x)
    df_test.insert(loc=0, column='class', value=test_y + 1)
    df_test.columns = ['class', 'X1', 'X2', 'X3']
    r_dataframe_test = pandas2ri.py2ri(df_test)

    plot3ddb(nnew, robjects.IntVector(train_y + 1), r_dataframe, 'radial', gamma, coef0, cost, tolerance, probability_,
             r_dataframe_test, True, title, "3d%s.png" % filename)

    # input('hello')


#     SPACE_SAMPLING_POINTS = train_x.shape[0]/3
#     X_MIN = int(min(train_x[:, 0].tolist()))
#     X_MAX = int(max(train_x[:, 0].tolist()))
#     Y_MIN = int(min(train_x[:, 1].tolist()))
#     Y_MAX = int(max(train_x[:, 1].tolist()))
#     Z_MIN = int(min(train_x[:, 1].tolist()))
#     Z_MAX = int(max(train_x[:, 1].tolist()))
#     print('X_MIN', X_MIN, 'X_MAX', X_MAX, 'Y_MIN', Y_MIN, 'Y_MAX', Y_MAX, 'Z_MIN', Z_MIN, 'Z_MAX', Z_MAX, 'SPACE_SAMPLING_POINTS', SPACE_SAMPLING_POINTS)
#     xx, yy, zz = np.meshgrid(np.linspace(X_MIN, X_MAX, SPACE_SAMPLING_POINTS),
#                              np.linspace(Y_MIN, Y_MAX, SPACE_SAMPLING_POINTS),
#                              np.linspace(Z_MIN, Z_MAX, SPACE_SAMPLING_POINTS))
#
#     Z = clf.decision_function(np.c_[yy.ravel(), xx.ravel(), zz.ravel()])
#     # if hasattr(clf, "decision_function"):
#     #     Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
#     # else:
#     #     Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])[:, 1]
#
#     Z = Z.reshape(xx.shape)
#     fig = plt.figure(figsize=(7, 6), dpi=100)
#     ax = fig.gca(projection='3d')
#
#
#     train_y_color = np.where(train_y == 0, 'dodgerblue', train_y)
#     test_y_color = np.where(test_y == 0, 'dodgerblue', test_y)
#
#     train_y_color = np.where(train_y_color == '1', 'orange', train_y_color)
#     test_y_color = np.where(test_y_color == '1', 'orange', test_y_color)
#
#     ax.scatter(train_x[:, 0], train_x[:, 1], train_x[:, 2], c=train_y_color, alpha=0.2)
#     ax.scatter(test_x[:, 0], test_x[:, 1], test_x[:, 2], edgecolors="red", c=test_y_color, alpha=1)
#
#     # ax.scatter(train_x[:, 0], train_x[:, 1], train_x[:, 2], c='red', alpha=1)
#     # ax.scatter(train_x[:, 1], train_x[:, 0], train_x[:, 2], c='blue', alpha=1)
#     # ax.scatter(test_x[:, 1], test_x[:, 2], test_x[:, 0], c='green', alpha=1)
#     # ax.scatter(train_x[:, 2], train_x[:, 0], train_x[:, 1], c='black', alpha=1)
#     # ax.scatter(test_x[:, 2], test_x[:, 1], test_x[:, 0], c='pink', alpha=1)
#
#     verts, faces = measure.marching_cubes_classic(Z)
#     verts = verts * [X_MAX - X_MIN, Y_MAX - Y_MIN, Z_MAX - Z_MIN] / SPACE_SAMPLING_POINTS
#     verts = np.add(verts, [X_MIN, Y_MIN, Z_MIN])
#
#     mesh = Poly3DCollection(verts[faces], facecolor='lightgray', alpha=0.1)
#     ax.add_collection3d(mesh)
#     ax.set_xlim((X_MIN, X_MAX))
#     ax.set_ylim((Y_MIN, Y_MAX))
#     ax.set_zlim((Z_MIN, Z_MAX))
#
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")
#     ax.legend([mpatches.Patch(color='gray', alpha=0.3),
#                Line2D([0], [0], marker='o', color='w', markerfacecolor='dodgerblue', markersize=10, alpha=0.9),
#                Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, alpha=0.9),
#                Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markersize=10, markeredgecolor='red',
#                       alpha=0.9)
#                ],
#               ["learned boundaries", "Class 0", "Class 1", "testing set"],
#               loc="lower left",
#               prop=matplotlib.font_manager.FontProperties(size=11))
#     ax.set_title(title)
#     fig.show()
#
#

def reduce_pls(output_dim, X_train, X_test, y_train, y_test):
    print("reduce pls...")
    clf = PLSRegression(n_components=output_dim)
    X_train = clf.fit_transform(X_train, y_train)[0]
    X_test = clf.fit_transform(X_test, y_test)[0]
    return X_train, X_test, y_train, y_test


def reduce_lda(output_dim, X_train, X_test, y_train, y_test):
    # lda implementation require 3 input class for 2d output and 4 input class for 3d output
    # if output_dim not in [1, 2, 3]:
    #     raise ValueError("available dimension for features reduction are 1, 2 and 3.")
    if output_dim == 3:
        X_train = np.vstack((X_train, np.array([np.zeros(X_train.shape[1]), np.ones(X_train.shape[1])])))
        y_train = np.append(y_train, (3, 4))
        X_test = np.vstack((X_test, np.array([np.zeros(X_test.shape[1]), np.ones(X_train.shape[1])])))
        y_test = np.append(y_test, (3, 4))
    if output_dim == 2:
        X_train = np.vstack((X_train, np.array([np.zeros(X_train.shape[1])])))
        y_train = np.append(y_train, 6)
        X_test = np.vstack((X_test, np.array([np.zeros(X_test.shape[1])])))
        y_test = np.append(y_test, 6)
    X_train = LDA(n_components=output_dim).fit_transform(X_train, y_train)
    X_test = LDA(n_components=output_dim).fit_transform(X_test, y_test)
    if output_dim != 1:
        X_train = X_train[0:-(output_dim - 1)]
        y_train = y_train[0:-(output_dim - 1)]
        X_test = X_test[0:-(output_dim - 1)]
        y_test = y_test[0:-(output_dim - 1)]

    return X_train, X_test, y_train, y_test


def reduce_pca(output_dim, X_train, X_test, y_train, y_test):
    if output_dim not in [2, 3]:
        raise ValueError("available dimension for features reduction are 2 and 3.")
    X_train = PCA(n_components=output_dim).fit_transform(X_train)
    X_test = PCA(n_components=output_dim).fit_transform(X_test)
    return X_train, X_test, y_train, y_test


def process_fold(n, X, y, train_index, test_index, dim_reduc=None):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

    if dim_reduc is None:
        return X, y, X_train, X_test, y_train, y_test

    if dim_reduc == 'LDA':
        X_train, X_test, y_train, y_test = reduce_lda(n, X_train, X_test, y_train, y_test)

    if dim_reduc == 'PLS':
        X_train, X_test, y_train, y_test = reduce_pls(n, X_train, X_test, y_train, y_test)

    if dim_reduc == 'PCA':
        X_train, X_test, y_train, y_test = reduce_pca(n, X_train, X_test, y_train, y_test)

    X_reduced = np.concatenate((X_train, X_test), axis=0)
    y_reduced = np.concatenate((y_train, y_test), axis=0)

    X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = X_reduced[train_index], X_reduced[test_index], \
                                                                       y_reduced[train_index], \
                                                                       y_reduced[test_index]
    return X_reduced, y_reduced, X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced


def process_fold2(n, X, y, X_t, y_t, dim_reduc=None):
    if dim_reduc is None:
        return X, y

    if dim_reduc == 'PLS':
        X_train, X_test, y_train, y_test = reduce_pls(n, X, X_t, y, y_t)

    if dim_reduc == 'LDA':
        X_train, X_test, y_train, y_test = reduce_lda(n, X, X_t, y, y_t)

    if dim_reduc == 'PCA':
        X_train, X_test, y_train, y_test = reduce_pca(n, X, X_t, y, y_t)

    X_reduced = np.concatenate((X_train, X_test), axis=0)
    y_reduced = np.concatenate((y_train, y_test), axis=0)

    return X_reduced, y_reduced, X_train, X_test, y_train, y_test


def get_proba(y_probas, y_pred):
    class_0 = []
    class_1 = []
    for i, item in enumerate(y_probas):
        if y_pred[i] == 0:
            class_0.append(item[0])
        if y_pred[i] == 1:
            class_1.append(item[1])

    class_0 = np.asarray(class_0)
    class_1 = np.asarray(class_1)

    return np.mean(class_0), np.mean(class_1)


def get_conf_interval(tprs, mean_fpr):
    confidence_lower = []
    confidence_upper = []
    df_tprs = pd.DataFrame(tprs, dtype=float)
    for column in df_tprs:
        scores = df_tprs[column].values.tolist()
        scores.sort()
        upper = np.percentile(scores, 95)
        confidence_upper.append(upper)
        lower = np.percentile(scores, 0.025)
        confidence_lower.append(lower)

    confidence_lower = np.asarray(confidence_lower)
    confidence_upper = np.asarray(confidence_upper)
    # confidence_upper = np.minimum(mean_tpr + std_tpr, 1)
    # confidence_lower = np.maximum(mean_tpr - std_tpr, 0)

    return confidence_lower, confidence_upper


def mean_confidence_interval(x):
    # boot_median = [np.median(np.random.choice(x, len(x))) for _ in range(iteration)]
    x.sort()
    lo_x_boot = np.percentile(x, 2.5)
    hi_x_boot = np.percentile(x, 97.5)
    print(lo_x_boot, hi_x_boot)
    return lo_x_boot, hi_x_boot


def plot_roc_range(ax, tprs, mean_fpr, aucs, out_dir, i, fig):
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='orange',
            label='Chance', alpha=1)

    mean_tpr = np.mean(tprs, axis=0)
    # mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    lo, hi = mean_confidence_interval(aucs)
    label = r'Mean ROC (Mean AUC = %0.2f, 95%% CI [%0.4f, %0.4f] )' % (mean_auc, lo, hi)
    if len(aucs) <= 2:
        label = r'Mean ROC (Mean AUC = %0.2f)' % mean_auc
    ax.plot(mean_fpr, mean_tpr, color='tab:blue',
            label=label,
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    confidence_lower, confidence_upper = get_conf_interval(tprs, mean_fpr)

    ax.fill_between(mean_fpr, confidence_lower, confidence_upper, color='tab:blue', alpha=.2)
                    #label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="Receiver operating characteristic iteration %d" % (i + 1))
    ax.legend(loc="lower right")
    # fig.show()
    path = "%s/roc_curve/" % (out_dir)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    final_path = '%s/%s' % (path, 'roc_%d.png' % i)
    final_path = final_path.replace('/', '\'').replace('\'', '\\').replace('\\', '/')
    print(final_path)
    fig.savefig(final_path)


def compute_model2(X, y, X_t, y_t, clf, dim=None, dim_reduc=None, clf_name=None, fname=None, outfname=None, days=None, resolution=None):
    # if clf_name not in ['SVM', 'MLP']:
    #     raise ValueError("available classifiers are SVM and MLP.")

    X_lda, y_lda, X_train, X_test, y_train, y_test = process_fold2(dim, X, y, X_t, y_t, dim_reduc=dim_reduc)
    # X_test, y_test = process_fold2(dim, X_t, y_t, dim_reduc=dim_reduc)
    X_lda = X_lda[:, :2] #if reduce to higher n slice to keep only first 2
    X_train = X_train[:, :2]
    X_test = X_test[:, :2]
    print("fit...")
    clf.fit(X_train, y_train)
    # clf = clf.best_estimator_
    # f_importances(clf.coef_.tolist()[0], [int(x) for x in range(0, clf.coef_.shape[1])], X_train)
    print("Best estimator found by grid search:")
    print(clf)
    y_pred = clf.predict(X_test)
    y_probas = clf.predict_proba(X_test)
    p_y_true, p_y_false = get_proba(y_probas, y_pred)
    print(p_y_true, p_y_false)
    acc = accuracy_score(y_test, y_pred)
    # mcc = matthews_corrcoef(y_test, y_pred)
    # print("MCC", mcc)
    skplt.metrics.plot_roc_curve(y_test, y_probas)
    plt.show()

    print(classification_report(y_test, y_pred))
    precision_false, precision_true, recall_false, recall_true, fscore_false, fscore_true, support_false, support_true = get_prec_recall_fscore_support(
        y_test, y_pred)

    if dim_reduc is None:
        return acc, precision_false, precision_true, recall_false, recall_true, fscore_false, fscore_true, support_false, support_true, clf_name

    if hasattr(clf, "hidden_layer_sizes"):
        clf_name = "%s%s%s" % (clf_name, str(clf.hidden_layer_sizes), fname.split('/')[-1])

    title = '%s-%s %dD \nacc=%.1f%%, p0=%d%%, p1=%d%%, r0=%d%%, r1=%d%%\ndataset: class0=%d; ' \
            'class1=%d\ntraining: class0=%d; class1=%d\ntesting: class0=%d; class1=%d\n' % (
                clf_name + '_' + fname.split('/')[-1], dim_reduc, dim,
                acc * 100, precision_false * 100, precision_true * 100, recall_false * 100, recall_true * 100,
                np.count_nonzero(y_lda == 0), np.count_nonzero(y_lda == 1),
                np.count_nonzero(y_train == 0), np.count_nonzero(y_train == 1),
                np.count_nonzero(y_test == 0), np.count_nonzero(y_test == 1))

    if dim == 3:
        plot_3D_decision_boundaries(X_lda, y_lda, X_test, y_test, title, clf, filename=outfname)

    if dim == 2:
        plot_2D_decision_boundaries(X_lda, y_lda, X_test, y_test, title, clf, filename=outfname, days=days,
                                    resolution=resolution)
        # plot_2D_decision_boundaries(X_lda, y_lda, X_test, title, clf, filename=outfname, days=days, resolution=resolution)
    if dim == 1:
        plot_2D_decision_boundaries(np.concatenate([X_test, X_lda]), np.concatenate([y_test, y_lda]), X_test, title,
                                    clf, filename=outfname)

    # skplt.metrics.plot_roc_curve(y_test, y_probas, title='ROC Curves\n%s' % title)
    # plt.show()
    fig, ax = plt.subplots()
    viz = plot_roc_curve(clf, X_lda, y_lda,
                         name='',
                         label='_Hidden',
                         alpha=0, lw=1, ax=ax)
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
    plot_roc_range(ax, tprs, mean_fpr, aucs, outfname, 0, fig)
    fig.clear()

    return acc, precision_false, precision_true, recall_false, recall_true, fscore_false, fscore_true, support_false, support_true, clf_name


def compute_model(X, y, train_index, test_index, i, clf, dim=None, dim_reduc=None, clf_name=None, outfname=None):
    # if clf_name not in ['SVM', 'MLP']:
    #     raise ValueError("available classifiers are SVM and MLP.")

    X_lda, y_lda, X_train, X_test, y_train, y_test = process_fold(dim, X, y, train_index, test_index,
                                                                  dim_reduc=dim_reduc)

    clf.fit(X_train, y_train)
    # f_importances(clf.coef_.tolist()[0], [int(x) for x in range(0, clf.coef_.shape[1])], X_train)

    print("Best estimator found by grid search:")
    print(clf)
    y_pred = clf.predict(X_test)
    y_probas = clf.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)

    # print(classification_report(y_test, y_pred))
    precision_false, precision_true, recall_false, recall_true, fscore_false, fscore_true, support_false, support_true = get_prec_recall_fscore_support(
        y_test, y_pred)

    if dim_reduc is None:
        return acc, precision_false, precision_true, recall_false, recall_true, fscore_false, fscore_true, support_false, support_true, clf_name

    if hasattr(clf, "hidden_layer_sizes"):
        clf_name = "%s%s" % (clf_name, str(clf.hidden_layer_sizes))

    title = '%s-%s %dD 10FCV\nfold_i=%d, acc=%.1f%%, p0=%d%%, p1=%d%%, r0=%d%%, r1=%d%%\ndataset: class0=%d;' \
            'class1=%d\ntraining: class0=%d; class1=%d\ntesting: class0=%d; class1=%d\n' % (
                clf_name, dim_reduc, dim, i,
                acc * 100, precision_false * 100, precision_true * 100, recall_false * 100, recall_true * 100,
                np.count_nonzero(y_lda == 0), np.count_nonzero(y_lda == 1),
                np.count_nonzero(y_train == 0), np.count_nonzero(y_train == 1),
                np.count_nonzero(y_test == 0), np.count_nonzero(y_test == 1))

    if dim == 3:
        plot_3D_decision_boundaries(X_lda, y_lda, X_test, y_test, title, clf, i=i)

    if dim == 2:
        plot_2D_decision_boundaries(X_lda, y_lda, X_test, y_test, title, clf, i=i, style2=False, days=0, resolution=resolution, filename=outfname)

    if dim == 1:
        plot_2D_decision_boundaries_(X_lda, y_lda, X_test, title, clf, i=i, filename=outfname, days=0, resolution=resolution)

    # skplt.metrics.plot_roc_curve(y_test, y_probas, title='ROC Curves\n%s' % title)
    # plt.show()
    simplified_results = {"accuracy": acc, "specificity": recall_false,
                          "recall": recall_score(y_test, y_pred, average='weighted'),
                          "precision": precision_score(y_test, y_pred, average='weighted'),
                          "f-score": f1_score(y_test, y_pred, average='weighted')}

    return acc, precision_false, precision_true, recall_false, recall_true, fscore_false, fscore_true, support_false, support_true, clf_name, simplified_results


def f_importances(coef, names, X_train):
    # imp = coef
    # imp, names = zip(*sorted(zip(imp, names)))
    # plt.barh(range(len(names)), imp, align='center')
    plt.locator_params(numticks=12)
    plt.plot(names, X_train[0])
    # plt.plot(names, coef, 'r--', X_train[0])
    # plt.yticks(range(len(names)), names)
    # plt.locator_params(axis='y', nbins=10)
    # plt.locator_params(axis='x', nbins=3)

    plt.show()


def process_fold_classic(n, X, y, dim_reduc=None, outfname=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    for i, trace in enumerate(X_train):
        label = y_train[i]
        path = outfname + '\\' + str(resolution) + '\\train\\'
        path_file = path + "%d_%d.png" % (label, i)
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        plt.clf()
        plt.bar(range(0, len(trace)), trace)
        plt.savefig(path_file, bbox_inches='tight')
        print(path_file)

    for i, trace in enumerate(X_test):
        label = y_train[i]
        path = outfname + '\\' + str(resolution) + '\\test\\'
        path_file = path + "%d_%d.png" % (label, i)
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        plt.clf()
        plt.bar(range(0, len(trace)), trace)
        plt.savefig(path_file, bbox_inches='tight')
        print(path_file)

    if dim_reduc is None:
        return X, y, X_train, X_test, y_train, y_test

    if dim_reduc == 'PLS':
        X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = reduce_pls(n, X_train, X_test, y_train, y_test)

    if dim_reduc == 'LDA':
        X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = reduce_lda(n, X_train, X_test, y_train, y_test)

    if dim_reduc == 'PCA':
        X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = reduce_pca(n, X_train, X_test, y_train, y_test)

    X_reduced = np.concatenate((X_train_reduced, X_test_reduced), axis=0)
    y_reduced = np.concatenate((y_train_reduced, y_test_reduced), axis=0)

    return X_reduced, y_reduced, X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced


def compute_model_classic_split(outfname, clf, clf_name, dim, X, y, dim_reduc, days, resolution):
    print("compute_model_classic_split...")
    X_lda, y_lda, X_train, X_test, y_train, y_test = process_fold_classic(2, X, y, dim_reduc="PLS", outfname=outfname)

    print(pd.DataFrame(X_train))
    print(pd.DataFrame(X_test))

    for i, trace in enumerate(X_train):
        label = y_train[i]
        path = outfname + '\\' + str(resolution) + '\\train\\reduced\\'
        path_file = path + "%d_%d.png" % (label, i)
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        plt.clf()
        plt.bar(range(0, len(trace)), trace)
        plt.savefig(path_file, bbox_inches='tight')
        print(path_file)

    for i, trace in enumerate(X_test):
        label = y_test[i]
        path = outfname + '\\' + str(resolution) + '\\test\\reduced\\'
        path_file = path + "%d_%d.png" % (label, i)
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        plt.clf()
        plt.bar(range(0, len(trace)), trace)
        plt.savefig(path_file, bbox_inches='tight')
        print(path_file)

    for i, trace in enumerate(X):
        label = y[i]
        path = outfname + '\\' + str(resolution) + '\\train\\X\\'
        path_file = path + "%d_%d.png" % (label, i)
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        plt.clf()
        plt.bar(range(0, len(trace)), trace)
        plt.savefig(path_file, bbox_inches='tight')
        print(path_file)


    clf.fit(X_train, y_train)

    print("Best estimator found by grid search:")
    print(clf)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    # print(classification_report(y_test, y_pred))
    precision_false, precision_true, recall_false, recall_true, fscore_false, fscore_true, support_false, support_true = get_prec_recall_fscore_support(
        y_test, y_pred)

    print((precision_false, precision_true, recall_false, recall_true, fscore_false, fscore_true, support_false, support_true))
    if dim_reduc is None:
        simplified_results = {"accuracy": acc, "specificity": recall_false,
                              "recall": recall_score(y_test, y_pred, average='weighted'),
                              "precision": precision_score(y_test, y_pred, average='weighted'),
                              "f-score": f1_score(y_test, y_pred, average='weighted')}
        return acc, precision_false, precision_true, recall_false, recall_true, fscore_false, fscore_true, support_false, support_true, clf_name, simplified_results

    title = '%s-%s %dD \nfold_i=%d, acc=%.1f%%, p0=%d%%, p1=%d%%, r0=%d%%, r1=%d%%\ndataset: class0=%d;' \
            'class1=%d\ntraining: class0=%d; class1=%d\ntesting: class0=%d; class1=%d\n' % (
                clf_name, dim_reduc, dim, 0,
                acc * 100, precision_false * 100, precision_true * 100, recall_false * 100, recall_true * 100,
                np.count_nonzero(y_lda == 0), np.count_nonzero(y_lda == 1),
                np.count_nonzero(y_train == 0), np.count_nonzero(y_train == 1),
                np.count_nonzero(y_test == 0), np.count_nonzero(y_test == 1))

    if dim == 2:
        plot_2D_decision_boundaries(X_lda, y_lda, X_test, y_test, title, clf, i=0, filename=outfname, days=days, resolution=resolution, style2=False)

    # skplt.metrics.plot_roc_curve(y_test, y_probas, title='ROC Curves\n%s' % title)
    # plt.show()
    simplified_results = {"accuracy": acc, "specificity": recall_false,
                          "recall": recall_score(y_test, y_pred, average='weighted'),
                          "precision": precision_score(y_test, y_pred, average='weighted'),
                          "f-score": f1_score(y_test, y_pred, average='weighted')}
    print(simplified_results)
    return simplified_results

from sklearn.linear_model import LinearRegression
def process(data_frame, fold=10, dim_reduc=None, clf_name=None, df2=None, fname=None, y_col='label', outfname=None, classic_split=None,
            days=None, resolution=None, df_original=None):
    if clf_name not in ['SVM', 'MLP', 'LREG', 'LDA']:
        raise ValueError('classifier %s is not available! available clf_name are MPL, LREG, SVM' % clf_name)
    scores_2d, scores_3d, scores_full = [], [], []
    precision_false_2d, precision_false_3d, precision_false_full = [], [], []
    precision_true_2d, precision_true_3d, precision_true_full = [], [], []
    recall_false_2d, recall_false_3d, recall_false_full = [], [], []
    recall_true_2d, recall_true_3d, recall_true_full = [], [], []
    fscore_false_2d, fscore_false_3d, fscore_false_full = [], [], []
    fscore_true_2d, fscore_true_3d, fscore_true_full = [], [], []
    support_false_2d, support_false_3d, support_false_full = [], [], []
    support_true_2d, support_true_3d, support_true_full = [], [], []
    simplified_results_2d, simplified_results_3d = [], []
    clf_name_full, clf_name_2d, clf_name_3d = '', '', ''

    X, y = process_data_frame(data_frame, y_col=y_col)
    #TODO animal wise split

    CV_iterator = []
    serials = list(set(df_original['serial']))
    chunks = [serials[x:x + 2] for x in range(0, len(serials), 2)]
    for i, chunck in enumerate(chunks):
        if len(chunck) == 2:
            train_indices = df_original[(df_original['serial'] == chunck[0]) | (df_original['serial'] == chunck[1])].index.values.astype(int)
            test_indices = df_original[(df_original['serial'] != chunck[0]) | (df_original['serial'] != chunck[1])].index.values.astype(int)
        else:
            train_indices = df_original[(df_original['serial'] == chunck[0])].index.values.astype(int)
            test_indices = df_original[(df_original['serial'] != chunck[0])].index.values.astype(int)
        CV_iterator.append((train_indices, test_indices))

    rkf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=int((datetime.now().microsecond) / 10))
    # rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=int((datetime.now().microsecond) / 10))
    # kf.get_n_splits(X)

    # clf = LDA()
    # param_grid = {'C': np.logspace(-6, -1, 10), 'gamma': np.logspace(-6, -1, 10)}
    # clf = GridSearchCV(SVC(kernel='rbf', probability=True), param_grid, cv=kf)
    clf = SVC(kernel='linear', probability=True)
    #
    # if clf_name == 'LREG':
    #     # param_grid = {'penalty': ['none', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    #     # clf = GridSearchCV(LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial'), param_grid, cv=kf)
    #     clf = LogisticRegression(C=1e10)
    #
    # if clf_name == 'MLP':
    #     param_grid = {'hidden_layer_sizes': [(5, 2), (5, 3), (5, 4), (5, 5), (4, 2), (4, 3), (4, 4), (2, 2), (3, 3)],
    #                   'alpha': [1e-8, 1e-8, 1e-10, 1e-11, 1e-12]}
    #     clf = GridSearchCV(MLPClassifier(solver='sgd', random_state=1), param_grid, cv=kf)

    if df2 is None:
        print("finding best estimator...")
        # clf.fit(X, y)
        # clf = clf.best_estimator_
        if classic_split:
            compute_model_classic_split(outfname, clf, clf_name, 2, X, y, dim_reduc, days, resolution)
        else:
            for i, (train_index, test_index) in enumerate(CV_iterator):
                print("progress %d/%d" % (i, fold))
                # acc_full, p_false_full, p_true_full, r_false_full, r_true_full, fs_false_full, fs_true_full, s_false_full, s_true_full, clf_name_full = compute_model(X, y, train_index, test_index, i, clf, clf_name=clf_name)
                # acc_3d, p_false_3d, p_true_3d, r_false_3d, r_true_3d, fs_false_3d, fs_true_3d, s_false_3d, s_true_3d, clf_name_3d, sr_3d = compute_model(
                #     X, y, train_index, test_index, i, clf, dim=3, dim_reduc=dim_reduc, clf_name=clf_name)
                acc_2d, p_false_2d, p_true_2d, r_false_2d, r_true_2d, fs_false_2d, fs_true_2d, s_false_2d, s_true_2d, clf_name_2d, sr_2d = compute_model(
                    X, y, train_index, test_index, i, clf, dim=2, dim_reduc=dim_reduc, clf_name=clf_name, outfname=outfname)
                # acc_1d, p_false_1d, p_true_1d, r_false_1d, r_true_1d, fs_false_1d, fs_true_1d, s_false_1d, s_true_1d, clf_name_1d, sr_1d = compute_model(
                #     X, y, train_index, test_index, i, clf, dim=1, dim_reduc=dim_reduc, clf_name=clf_name)

                # scores_full.append(acc_full)
                # precision_false_full.append(p_false_full)
                # precision_true_full.append(p_true_full)
                # recall_false_full.append(r_false_full)
                # recall_true_full.append(r_true_full)
                # fscore_false_full.append(fs_false_full)
                # fscore_true_full.append(fs_true_full)
                # support_false_full.append(s_false_full)
                # support_true_full.append(s_true_full)


                scores_2d.append(acc_2d)
                precision_false_2d.append(p_false_2d)
                precision_true_2d.append(p_true_2d)
                recall_false_2d.append(r_false_2d)
                recall_true_2d.append(r_true_2d)
                fscore_false_2d.append(fs_false_2d)
                fscore_true_2d.append(fs_true_2d)
                support_false_2d.append(s_false_2d)
                support_true_2d.append(s_true_2d)
                simplified_results_2d.append(sr_2d)

                # scores_3d.append(acc_3d)
                # precision_false_3d.append(p_false_3d)
                # precision_true_3d.append(p_true_3d)
                # recall_false_3d.append(r_false_3d)
                # recall_true_3d.append(r_true_3d)
                # fscore_false_3d.append(fs_false_3d)
                # fscore_true_3d.append(fs_true_3d)
                # support_false_3d.append(s_false_3d)
                # support_true_3d.append(s_true_3d)
                # simplified_results_3d.append(sr_3d)
            print("svc %d fold cross validation 2d is %f, 3d is %s." % (
                fold, float(np.mean(scores_2d)), float(np.mean(scores_3d))))
            # print(float(np.mean(acc_val_list)))
            result = {
                'fold': fold,
                '2d_reduced': {
                    'clf_name': clf_name_2d,
                    'accuracy': float(np.mean(scores_2d)),
                    'precision_true_2d': float(np.mean(precision_true_2d)),
                    'precision_false_2d': np.mean(precision_false_2d),
                    'recall_true_2d': float(np.mean(recall_true_2d)),
                    'recall_false_2d': np.mean(recall_false_2d),
                    'fscore_true_2d': float(np.mean(fscore_true_2d)),
                    'fscore_false_2d': float(np.mean(fscore_false_2d)),
                    'support_true_2d': np.mean(support_true_2d),
                    'support_false_2d': np.mean(support_false_2d),
                }
                ,
                '3d_reduced': {
                    'clf_name': clf_name_3d,
                    'accuracy': float(np.mean(scores_3d)),
                    'precision_true_3d': float(np.mean(precision_true_3d)),
                    'precision_false_3d': np.mean(precision_false_3d),
                    'recall_true_3d': float(np.mean(recall_true_3d)),
                    'recall_false_3d': np.mean(recall_false_3d),
                    'fscore_true_3d': float(np.mean(fscore_true_3d)),
                    'fscore_false_3d': float(np.mean(fscore_false_3d)),
                    'support_true_3d': np.mean(support_true_3d),
                    'support_false_3d': np.mean(support_false_3d),
                }
                ,
                'simplified_results': {
                    'simplified_results_2d': simplified_results_2d,
                    'simplified_results_3d': simplified_results_3d
                }
            }
            print(result)
            return result
    else:
        # clf = LinearRegression(probability=True)
        # clf.fit(X, y)
        # clf = clf.best_estimator_
        clf = SVC(kernel='linear', probability=True,  C=0.00001)
        X_t, y_t = process_data_frame(df2, y_col=y_col)
        # acc_3d, p_false_3d, p_true_3d, r_false_3d, r_true_3d, fs_false_3d, fs_true_3d, s_false_3d, s_true_3d,\
        # clf_name_3d = compute_model2(X, y, X_t, y_t, clf, dim=3, dim_reduc=dim_reduc, clf_name=clf_name, fname=fname)
        acc_2d, p_false_2d, p_true_2d, r_false_2d, r_true_2d, fs_false_2d, fs_true_2d, s_false_2d, s_true_2d, clf_name_2d = compute_model2(
            X, y, X_t, y_t, clf, dim=2, dim_reduc=dim_reduc, clf_name=clf_name, fname=fname, outfname=outfname, days=days, resolution=resolution)
        # acc_1d, p_false_1d, p_true_1d, r_false_1d, r_true_1d, fs_false_1d, fs_true_1d, s_false_1d, s_true_1d, clf_name_1d = compute_model2(
        #     X, y, X_t, y_t, clf, dim=1, dim_reduc=dim_reduc, clf_name=clf_name, fname=fname)

        # print(acc_3d, p_false_3d, p_true_3d, r_false_3d, r_true_3d, fs_false_3d, fs_true_3d, s_false_3d, s_true_3d, clf_name_3d)
        print(acc_2d, p_false_2d, p_true_2d, r_false_2d, r_true_2d, fs_false_2d, fs_true_2d, s_false_2d, s_true_2d,
              clf_name_2d)


def find_type_for_mem_opt(df):
    data_col_n = df.iloc[[0]].size
    type_dict = {}
    for n, i in enumerate(range(0, data_col_n)):
        if n < (data_col_n - META_DATA_LENGTH):
            type_dict[str(i)] = np.float16
        else:
            type_dict[str(i)] = np.str
    del df
    type_dict[str(data_col_n-1)] = np.int
    type_dict[str(data_col_n-2)] = np.int
    type_dict[str(data_col_n-3)] = np.int
    type_dict[str(data_col_n-8)] = np.int
    type_dict[str(data_col_n - 9)] = np.int
    type_dict[str(data_col_n - 10)] = np.int
    type_dict[str(data_col_n - 11)] = np.int
    type_dict[str(data_col_n - 15)] = np.int
    return type_dict


def load_df_from_datasets(fname, label_col):
    df = pd.read_csv(fname, nrows=5, sep=",", header=None)

    type_dict = find_type_for_mem_opt(df)
    data_frame = pd.read_csv(fname, sep=",", header=None, low_memory=False, dtype=type_dict)
    sample_count = data_frame.shape[1]
    hearder = [str(n) for n in range(0, sample_count)]
    hearder[-19] = "label"
    hearder[-18] = "elem_in_row"
    hearder[-17] = "date1"
    hearder[-16] = "date2"
    hearder[-15] = "serial"
    hearder[-14] = "famacha_score"
    hearder[-13] = "previous_famacha_score"
    hearder[-12] = "previous_famacha_score2"
    hearder[-11] = "previous_famacha_score3"
    hearder[-10] = "previous_famacha_score4"

    hearder[-9] = "dtf1"
    hearder[-8] = "dtf2"
    hearder[-7] = "dtf3"
    hearder[-6] = "dtf4"
    hearder[-5] = "dtf5"

    hearder[-4] = "nd1"
    hearder[-3] = "nd2"
    hearder[-2] = "nd3"
    hearder[-1] = "nd4"

    data_frame.columns = hearder
    print(data_frame)
    data_frame["famacha_score"] = pd.to_numeric(data_frame["famacha_score"])
    data_frame["previous_famacha_score"] = pd.to_numeric(data_frame["previous_famacha_score"])
    data_frame["previous_famacha_score2"] = pd.to_numeric(data_frame["previous_famacha_score2"])
    data_frame["previous_famacha_score3"] = pd.to_numeric(data_frame["previous_famacha_score3"])
    # data_frame = data_frame[data_frame.famacha_score > 0]
    # data_frame = data_frame[data_frame.previous_famacha_score > 0]
    # data_frame = data_frame[data_frame.previous_famacha_score2 > 0]
    # data_frame = data_frame[data_frame.previous_famacha_score3 > 0]

    grouped_by_id = data_frame.copy()
    # for region, df_serial in data_frame.groupby('serial'):
    #     print(df_serial)
    #     grouped_by_id.append(df_serial)
    data_frame_original = data_frame.copy()
    cols_to_keep = hearder[:-META_DATA_LENGTH]
    cols_to_keep.append(label_col)
    data_frame = data_frame[cols_to_keep]
    data_frame = shuffle(data_frame)
    # row_to_delete = []
    # for index, row in data_frame.iterrows():
    #     row = list(row)
    #     a, b = np.unique(row, return_counts=True)
    #     most_abundant_value = a[b.argmax()]
    #     occurance = b.max()
    #     if abs(float(most_abundant_value)) != 0 and 2000 < occurance < 50:
    #         row_to_delete.append(index)
    #         continue
    #     # plt.plot(row)
    #     # plt.show()
    # data_frame.drop(data_frame.index[row_to_delete])
    # print(data_frame)
    return data_frame_original, data_frame, cols_to_keep, grouped_by_id


def start(fname1=None, fname2=None, half_period_split=False, label_col='label', outfname=None, classic_split=None, days=None, resolution=None):
    if fname2 is not None:
        print("use different two different dataset for training and testing.\n"
              "training set:%s\n testing set:%s" % (fname1, fname2))
        _, df1, cols_to_keep, grouped_by_id = load_df_from_datasets(fname1, label_col)
        _, df2, cols_to_keep_2, grouped_by_id = load_df_from_datasets(fname2, label_col)
        df1 = df1[cols_to_keep]
        df2 = df2[cols_to_keep]
        print(df1.shape)
        print(df2.shape)
        print("data loading finished.")
        try:
            class_true_count = df1[label_col].value_counts().to_dict()[True]
            class_false_count = df1[label_col].value_counts().to_dict()[False]
        except KeyError as e:
            print(e)
        print("class_true_count=%d and class_false_count=%d" % (class_true_count, class_false_count))
        print("current_file is", fname1)
        # exit(0)
        process(df1, df2=df2, dim_reduc='LDA', clf_name='SVM', fname=fname1, outfname=outfname, days=days, resolution=resolution)
        return

    print("loading dataset...")
    data_frame, _, cols_to_keep, grouped_by_id = load_df_from_datasets(fname1, label_col)
    list_of_df = [g for _, g in data_frame.groupby(['famacha_score'])]

    if half_period_split:
        data_frame['date1'] = pd.to_datetime(data_frame['date1'], dayfirst=True)
        data_frame['date2'] = pd.to_datetime(data_frame['date2'], dayfirst=True)
        data_frame = data_frame.sort_values('date1', ascending=True)
        print(data_frame)
        nrows = int(data_frame.shape[0] / 2)
        print(nrows)

        print('data_frame:%s %s' % (str(data_frame["date1"].iloc[0]).split(' ')[0], str(data_frame["date1"].iloc[-1]).split(' ')[0]))

        # if 'delmas' in fname1:
        #     data_frame = data_frame.loc[(data_frame['date1'] >= datetime(2015, 4, 1))]
        #     data_frame = data_frame.loc[(data_frame['date1'] <= datetime(2016, 4, 1))]
        #
        # if 'cedara' in fname1:
        #     data_frame = data_frame.loc[(data_frame['date1'] >= datetime(2012, 4, 1))]
        #     data_frame = data_frame.loc[(data_frame['date1'] <= datetime(2013, 4, 1))]

        df1 = data_frame[:nrows]
        df2 = data_frame[nrows:]
        print(df1)
        print(df2)
        print(
            'df1:%s %s\ndf2:%s %s' % (str(df1["date1"].iloc[0]).split(' ')[0], str(df1["date1"].iloc[-1]).split(' ')[0],
                                      str(df2["date1"].iloc[0]).split(' ')[0],
                                      str(df2["date1"].iloc[-1]).split(' ')[0]))
        df1 = df1[cols_to_keep]
        df1 = df1.sample(frac=1).reset_index(drop=True)
        df1 = df1.fillna(-1)
        df1 = shuffle(df1)

        df2 = df2[cols_to_keep]
        df2 = df2.sample(frac=1).reset_index(drop=True)
        df2 = df2.fillna(-1)
        df2 = shuffle(df2)

        class_1_count = data_frame['label'].value_counts().to_dict()[True]
        class_2_count = data_frame['label'].value_counts().to_dict()[False]
        print("class_true_count=%d and class_false_count=%d" % (class_1_count, class_2_count))
        process(df1, df2=df2, dim_reduc='LDA', clf_name='SVM', fname=fname1, outfname=outfname+'12', y_col=label_col, days=days, resolution=resolution)
        process(df2, df2=df1, dim_reduc='LDA', clf_name='SVM', fname=fname1, outfname=outfname+'21', y_col=label_col, days=days, resolution=resolution)
    else:
        data_frame = data_frame.sample(frac=1).reset_index(drop=True)
        data_frame = data_frame.fillna(-1)
        process(data_frame, dim_reduc='LDA', clf_name='LREG', y_col=label_col, outfname=outfname, classic_split=classic_split, days=days, resolution=resolution, df_original=grouped_by_id)


if __name__ == '__main__':



    # dir = MAIN_DIR + "10min_sld_6_dbt6_cedara_70091100056/training_sets/"
    # # os.chdir(dir)

    # start(fname1=MAIN_DIR + "10min_sld_0_dbt7_delmas_70101200027/training_sets/cwt_humidity_temperature_.data", classic_split=True, outfname="delmas_cwt_humidity_temperature_")
    # start(fname1=MAIN_DIR + "10min_sld_0_dbt7_delmas_70101200027/training_sets/cwt_div.data", classic_split=True, outfname="delmas_cwt_div")
    #
    # exit()
    #
    resolutions = ["10min"]
    days = [7]
    # try:
    #     # shutil.rmtree("cross_farm", ignore_errors=True)
    # except (OSError, FileNotFoundError) as e:
    #     print(e)
    # try:
    #     shutil.rmtree("half_split", ignore_errors=True)
    # except (OSError, FileNotFoundError) as e:
    #     print(e)
    for resolution in resolutions:
        for day in days:
            # start(fname1=MAIN_DIR + "%s_sld_0_dbt%d_delmas_70101200027/training_sets/cwt_.data" % (resolution, day),
            #       half_period_split=True,
            #       outfname="herd_lev_var\\delmas_half", days=day, resolution=resolution)
            # start(fname1=MAIN_DIR + "%s_sld_0_dbt%d_delmas_70101200027/training_sets/cwt_div.data" % (resolution, day),
            #       half_period_split=True,
            #       days=day, resolution=resolution,
            #       outfname="herd_lev_var\\cwt_div")
            # start(fname1=MAIN_DIR + "%s_sld_0_dbt%d_cedara_70091100056/training_sets/cwt_.data" % (resolution, day),
            #       half_period_split=True,
            #       days=day, resolution=resolution,
            #       outfname="half\\cedara")

            # start(fname1=MAIN_DIR + "%s_sld_0_dbt%d_delmas_70101200027/training_sets/cwt_.data" % (resolution, day), half_period_split=True,
            #       outfname="half_split\\delmas_half", days=day, resolution=resolution)
            #
            # start(fname1=MAIN_DIR + "%s_sld_0_dbt%d_cedara_70091100056/0_00_1_00_0_00/training_sets/cwt_.data" % (resolution, day), half_period_split=True,
            #       outfname="half_split\\cedara_half", days=day, resolution=resolution)
            
            start(fname1=MAIN_DIR + "%s_sld_0_dbt%d_delmas_70101200027/0_00_0_00_0_00/training_sets/cwt_.data" % (resolution, day),
                  outfname="PLS",
                  days=day, resolution=resolution, classic_split=True)

            # start(fname2=MAIN_DIR + "%s_sld_0_dbt%d_delmas_70101200027/training_sets/cwt_.data" % (resolution, day),
            #       fname1=MAIN_DIR + "%s_sld_0_dbt%d_cedara_70091100056/training_sets/cwt_.data" % (resolution, day),
            #       outfname="cross_farm\\trained_on_cedara_test_on_delmas",
            #       days=day, resolution=resolution)

            # start(fname2=MAIN_DIR + "%s_sld_0_dbt%d_delmas_70101200027/training_sets/cwt_.data" % (resolution, day),
            #       fname1=MAIN_DIR + "%s_sld_0_dbt%d_cedara_70091100056/training_sets/cwt_.data" % (resolution, day),
            #       outfname="cross_farm\\trained_on_cedara_test_on_delmas",
            #       days=day, resolution=resolution)
            # 




    # start(fname1=MAIN_DIR + "10min_sld_0_dbt7_delmas_70101200027/training_sets/cwt_.data", classic_split=True,
    #       outfname="delmas_classic")
    # start(fname1=MAIN_DIR + "10min_sld_0_dbt7_cedara_70091100056/training_sets/cwt_.data", classic_split=True,
    #       outfname="cedara_classic")

    # start(fname1=dir+"cwt_.data", half_period_split=False, outfname="cedara_cwt_famacha_score", label_col="famacha_score")
    #


    # dir = MAIN_DIR + "10min_sld_6_dbt6_cedara_70091100056/training_sets/"
    # # os.chdir(dir)
    # start(fname1=dir+"cwt_.data", half_period_split=True, outfname="cedara_cwt")
    # start(fname1=dir+"cwt_div.data", half_period_split=True, outfname="cedara_cwt_div")

    # dir = MAIN_DIR + "10min_sld_0_dbt6_delmas_70101200027/training_sets/"
    # os.chdir(dir)
    # start(fname1=dir+"cwt_humidity_temperature_.data", half_period_split=True, outfname="delmas_cwt_humidity_temperature_")
    # start(fname1=dir+"cwt_humidity_temperature_div.data", half_period_split=True, outfname="delmas_cwt_humidity_temperature_div")
