import math
import warnings

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
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.utils import shuffle
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as py
import plotly.tools as tls
from IPython.display import display
import plotly
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

import os
os.environ['R_HOME'] = 'C:\Program Files\R\R-3.6.0' #path to your R installation
os.environ['R_USER'] = 'C:\\Users\\fo18103\\AppData\\Local\Continuum\\anaconda3\Lib\site-packages\\rpy2' #path depends on where you installed Python. Mine is the Anaconda distribution

import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
R = robjects.r

e1071 = importr('e1071')
rgl = importr('rgl')
misc3d = importr('misc3d')
plot3D = importr('plot3D')
plot3Drgl = importr('plot3Drgl')

print(rpy2.__version__)


warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)




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


def process_data_frame(data_frame):
    data_frame = data_frame.fillna(-1)
    cwt_shape = data_frame[data_frame.columns[0:2]].values
    X = data_frame[data_frame.columns[2:data_frame.shape[1] - 1]].values

    # for x in X:
    #     cwt = x.reshape(cwt_shape[0])
    #     print(cwt)
    #     create_cwt_graph(cwt, cwt_shape[0][1])

    X = normalize(X)
    X = preprocessing.MinMaxScaler().fit_transform(X)
    # print(X.shape, X)
    # print(DataFrame.from_records(X))
    y = data_frame["class"].values.flatten()
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


def plot_2D_decision_boundaries(X, y, X_test, title, clf, i=0):
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
    # plt.savefig('%d' % i)
    plt.show()
    plt.close()


def plot_3D_decision_boundaries(train_x, train_y, test_x, test_y, title, clf, i=0):
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
    df.insert(loc=0, column='group', value=train_y+1)
    df.columns = ['group', 'X1', 'X2', 'X3']
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    r_dataframe = pandas2ri.py2ri(df)

    df_test = pd.DataFrame(test_x)
    df_test.insert(loc=0, column='class', value=test_y+1)
    df_test.columns = ['class', 'X1', 'X2', 'X3']
    r_dataframe_test = pandas2ri.py2ri(df_test)

    plot3ddb(nnew, robjects.IntVector(train_y+1), r_dataframe, 'radial', gamma, coef0, cost, tolerance, probability_,
             r_dataframe_test, True, title, 'E:/downloads/%d.png' % i)

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
def reduce_lda(output_dim, X_train, X_test, y_train, y_test):
    #lda implementation require 3 input class for 2d output and 4 input class for 3d output
    if output_dim not in [2, 3]:
        raise ValueError("available dimension for features reduction are 2 and 3.")
    if output_dim == 3:
        X_train = np.vstack((X_train, np.array([np.zeros(X_train.shape[1]), np.ones(X_train.shape[1])])))
        y_train = np.append(y_train, (3, 4))
        X_test = np.vstack((X_test, np.array([np.zeros(X_test.shape[1]), np.ones(X_train.shape[1])])))
        y_test = np.append(y_test, (3, 4))
    if output_dim == 2:
        X_train = np.vstack((X_train, np.array([np.zeros(X_train.shape[1])])))
        y_train = np.append(y_train, 3)
        X_test = np.vstack((X_test, np.array([np.zeros(X_test.shape[1])])))
        y_test = np.append(y_test, 3)
    X_train = LDA(n_components=output_dim).fit_transform(X_train, y_train)
    X_test = LDA(n_components=output_dim).fit_transform(X_test, y_test)
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
    
    if dim_reduc == 'PCA':
        X_train, X_test, y_train, y_test = reduce_pca(n, X_train, X_test, y_train, y_test)

    X_reduced = np.concatenate((X_train, X_test), axis=0)
    y_reduced = np.concatenate((y_train, y_test), axis=0)

    X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = X_reduced[train_index], X_reduced[test_index], y_reduced[train_index], \
                                                       y_reduced[test_index]
    return X_reduced, y_reduced, X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced


def process_fold2(n, X, y, dim_reduc=None):

    if dim_reduc is None:
        return X, y

    if dim_reduc == 'LDA':
        X_train, X_test, y_train, y_test = reduce_lda(n, X, X, y, y)

    if dim_reduc == 'PCA':
        X_train, X_test, y_train, y_test = reduce_pca(n, X, X, y, y)

    X_reduced = np.concatenate((X_train, X_test), axis=0)
    y_reduced = np.concatenate((y_train, y_test), axis=0)

    return X_reduced, y_reduced

def compute_model2(X, y, X_t, y_t, clf, dim=None, dim_reduc=None, clf_name=None):
    if clf_name not in ['SVC', 'MLP']:
        raise ValueError("available classifiers are SVC and MLP.")

    X_lda, y_lda = process_fold2(dim, X, y, dim_reduc=dim_reduc)
    X_test, y_test = process_fold2(dim, X_t, y_t, dim_reduc=dim_reduc)

    print("fit...")
    clf.fit(X_lda, y_lda)
    # f_importances(clf.coef_.tolist()[0], [int(x) for x in range(0, clf.coef_.shape[1])], X_train)

    print("Best estimator found by grid search:")
    print(clf)
    y_pred = clf.predict(X_test)
    y_probas = clf.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    print("MCC", mcc)

    print(classification_report(y_test, y_pred))
    precision_false, precision_true, recall_false, recall_true, fscore_false, fscore_true, support_false, support_true = get_prec_recall_fscore_support(
        y_test, y_pred)

    if dim_reduc is None:
        return acc, precision_false, precision_true, recall_false, recall_true, fscore_false, fscore_true, support_false, support_true, clf_name

    if hasattr(clf, "hidden_layer_sizes"):
        clf_name = "%s%s" % (clf_name, str(clf.hidden_layer_sizes))

    title = '%s-%s %dD 10FCV\nfold_i=%d, acc=%.1f%%, p0=%d%%, p1=%d%%, r0=%d%%, r1=%d%%\ndataset: class0=%d;' \
            'class1=%d\ntraining: class0=%d; class1=%d\ntesting: class0=%d; class1=%d\n' % (
                clf_name, dim_reduc, dim, 0,
                acc * 100, precision_false * 100, precision_true * 100, recall_false * 100, recall_true * 100,
                np.count_nonzero(y_lda == 0), np.count_nonzero(y_lda == 1),
                np.count_nonzero(y_t == 0), np.count_nonzero(y_t == 1),
                np.count_nonzero(y_test == 0), np.count_nonzero(y_test == 1))

    # if dim == 3:
    #     plot_3D_decision_boundaries(X_lda, y_lda, X_test, y_test, title, clf, i=0)
    #
    if dim == 2:
        plot_2D_decision_boundaries(np.concatenate([X_test, X_lda]), np.concatenate([y_test, y_lda]), X_test, title, clf, i=0)


    # skplt.metrics.plot_roc_curve(y_test, y_probas, title='ROC Curves\n%s' % title)
    # plt.show()

    return acc, precision_false, precision_true, recall_false, recall_true, fscore_false, fscore_true, support_false, support_true, clf_name


def compute_model(X, y, train_index, test_index, i, clf, dim=None, dim_reduc=None, clf_name=None):
    if clf_name not in ['SVC', 'MLP']:
        raise ValueError("available classifiers are SVC and MLP.")

    X_lda, y_lda, X_train, X_test, y_train, y_test = process_fold(dim, X, y, train_index, test_index, dim_reduc=dim_reduc)

    clf.fit(X, y)
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
        plot_2D_decision_boundaries(X_lda, y_lda, X_test, title, clf, i=i)


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


def process(data_frame, fold=10, dim_reduc=None, clf_name=None, df2=None):
    X, y = process_data_frame(data_frame)
    X_t, y_t = process_data_frame(df2)

    y = y.astype(int)
    kf = StratifiedKFold(n_splits=fold, random_state=None, shuffle=True)
    kf.get_n_splits(X)
    
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
    if clf_name == 'SVC':
        param_grid = {'C': np.logspace(-6, -1, 10), 'gamma': np.logspace(-6, -1, 10)}
        clf = GridSearchCV(SVC(kernel='linear', probability=True), param_grid, cv=kf)
        # clf = LDA()
        # clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')

    if clf_name == 'MLP':
        param_grid = {'hidden_layer_sizes': [(5, 2), (5, 3), (5, 4), (5, 5), (4, 2), (4, 3), (4, 4), (2, 2), (3, 3)],
                      'alpha': [1e-8, 1e-8, 1e-10, 1e-11, 1e-12]}
        clf = GridSearchCV(MLPClassifier(solver='sgd', random_state=1), param_grid, cv=kf)

    # acc_3d, p_false_3d, p_true_3d, r_false_3d, r_true_3d, fs_false_3d, fs_true_3d, s_false_3d, s_true_3d, clf_name_3d = compute_model2(X, y, X_t, y_t, clf, dim=3, dim_reduc=dim_reduc, clf_name=clf_name)
    acc_2d, p_false_2d, p_true_2d, r_false_2d, r_true_2d, fs_false_2d, fs_true_2d, s_false_2d, s_true_2d, clf_name_2d = compute_model2(X, y, X_t, y_t, clf, dim=2, dim_reduc=dim_reduc, clf_name=clf_name)

    # print(acc_3d, p_false_3d, p_true_3d, r_false_3d, r_true_3d, fs_false_3d, fs_true_3d, s_false_3d, s_true_3d, clf_name_3d)
    print(acc_2d, p_false_2d, p_true_2d, r_false_2d, r_true_2d, fs_false_2d, fs_true_2d, s_false_2d, s_true_2d, clf_name_2d)

    exit()
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        print("%d/%d" % (i, fold))
        # acc_full, p_false_full, p_true_full, r_false_full, r_true_full, fs_false_full, fs_true_full, s_false_full, s_true_full, clf_name_full = compute_model(X, y, train_index, test_index, i, clf, clf_name=clf_name)
        acc_3d, p_false_3d, p_true_3d, r_false_3d, r_true_3d, fs_false_3d, fs_true_3d, s_false_3d, s_true_3d, clf_name_3d, sr_3d = compute_model(X, y, train_index, test_index, i, clf, dim=3, dim_reduc=dim_reduc, clf_name=clf_name)
        acc_2d, p_false_2d, p_true_2d, r_false_2d, r_true_2d, fs_false_2d, fs_true_2d, s_false_2d, s_true_2d, clf_name_2d, sr_2d = compute_model(X, y, train_index, test_index, i, clf, dim=2, dim_reduc=dim_reduc, clf_name=clf_name)

        # scores_full.append(acc_full)
        # precision_false_full.append(p_false_full)
        # precision_true_full.append(p_true_full)
        # recall_false_full.append(r_false_full)
        # recall_true_full.append(r_true_full)
        # fscore_false_full.append(fs_false_full)
        # fscore_true_full.append(fs_true_full)
        # support_false_full.append(s_false_full)
        # support_true_full.append(s_true_full)
        simplified_results_2d.append(sr_2d)
        simplified_results_3d.append(sr_3d)

        scores_2d.append(acc_2d)
        precision_false_2d.append(p_false_2d)
        precision_true_2d.append(p_true_2d)
        recall_false_2d.append(r_false_2d)
        recall_true_2d.append(r_true_2d)
        fscore_false_2d.append(fs_false_2d)
        fscore_true_2d.append(fs_true_2d)
        support_false_2d.append(s_false_2d)
        support_true_2d.append(s_true_2d)

        scores_3d.append(acc_3d)
        precision_false_3d.append(p_false_3d)
        precision_true_3d.append(p_true_3d)
        recall_false_3d.append(r_false_3d)
        recall_true_3d.append(r_true_3d)
        fscore_false_3d.append(fs_false_3d)
        fscore_true_3d.append(fs_true_3d)
        support_false_3d.append(s_false_3d)
        support_true_3d.append(s_true_3d)

    print("svc %d fold cross validation 2d is %f, 3d is %s." % (fold, float(np.mean(scores_2d)), float(np.mean(scores_3d))))
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


def start(fname=''):
    print("loading dataset...")
    # print(fname)

    df = pd.read_csv(fname, nrows=1, sep=",", header=None)
    print(df)
    data_col_n = df.iloc[[0]].size
    type_dict = {}
    for n, i in enumerate(range(0, data_col_n)):
        if n < (data_col_n - 5):
            type_dict[str(i)] = np.float16
        else:
            type_dict[str(i)] = np.str

    data_frame = pd.read_csv(fname, sep=",", header=None, dtype=type_dict)
    # data_frame = pd.concat(tfr, ignore_index=True)
    print(data_frame)
    sample_count = data_frame.shape[1]

    hearder = [str(n) for n in range(0, sample_count)]
    hearder[-5] = "class"
    hearder[-4] = "elem_in_row"
    hearder[-3] = "date1"
    hearder[-2] = "date2"
    hearder[-1] = "serial"
    data_frame.columns = hearder

    # data_frame = data_frame.loc[:, :'class']
    # np.random.seed(0)
    # data_frame = data_frame.sample(frac=1).reset_index(drop=True)
    # data_frame = data_frame.fillna(-1)
    # data_frame = shuffle(data_frame)
    # process(data_frame, dim_reduc='LDA', clf_name='SVC')

    data_frame['date1'] = pd.to_datetime(data_frame['date1'], dayfirst=True)
    data_frame['date2'] = pd.to_datetime(data_frame['date2'], dayfirst=True)
    data_frame = data_frame.sort_values('date1', ascending=True)
    print(data_frame)
    nrows = int(data_frame.shape[0]/2)
    print(nrows)
    df1 = data_frame[:nrows]
    df2 = data_frame[nrows:]
    print(df1)
    print(df2)
    print('df1:%s %s\ndf2:%s %s' % (str(df1["date1"].iloc[0]).split(' ')[0], str(df1["date1"].iloc[-1]).split(' ')[0],
                                    str(df2["date1"].iloc[0]).split(' ')[0], str(df2["date1"].iloc[-1]).split(' ')[0]))
    df1 = df1.loc[:, :'class']
    df1 = df1.sample(frac=1).reset_index(drop=True)
    df1 = df1.fillna(-1)
    df1 = shuffle(df1)

    df2 = df2.loc[:, :'class']
    df2 = df2.sample(frac=1).reset_index(drop=True)
    df2 = df2.fillna(-1)
    df2 = shuffle(df2)

    class_1_count = data_frame['class'].value_counts().to_dict()[True]
    class_2_count = data_frame['class'].value_counts().to_dict()[False]
    print("class_true_count=%d and class_false_count=%d" % (class_1_count, class_2_count))
    # process(df2, dim_reduc='LDA', clf_name='SVC', df2=df1)
    process(df2, dim_reduc='LDA', clf_name='SVC', df2=df1)

if __name__ == '__main__':
    # start()
    start(
        fname="C:/Users/fo18103/PycharmProjects/prediction_of_helminths_infection/training_data_generator_and_ml_classifier/src/resolution_10min_days_5_div/training_sets/cwt_.data")
    # start(fname="C:/Users/fo18103/PycharmProjects/training_data_generator/src/resolution_10min_days_6/training_sets/cwt_.data")
