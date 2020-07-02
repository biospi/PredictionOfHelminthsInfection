import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def plot(X, y, title):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    # plot support vectors
    if hasattr(clf, 'support_vectors_'):
        ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
                   linewidth=1, facecolors='none', edgecolors='k')
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    # X = np.array([[1, 2], [1, -2], [-1, -2]])
    X = np.array([[-1, 2], [1, 2], [-1, -2]])
    y = np.array([-1, -1, 1])

    for clf in [LDA(), SVC(kernel='linear')]:
        clf.fit(X, y)
        plot(X, y, "%s [%.1f %.1f]" % (str(type(clf)).split('.')[-1].replace('\'>', ''), clf.coef_[0][0], clf.coef_[0][1]))
        print(clf, clf.coef_)