import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_circles

# Synthetic data
X, y = make_circles(noise=0.3, factor=0.5, random_state=1)

from sklearn.svm import SVC

classifiers = [        # Three different SVM classifiers with difference values of gamma.
    SVC(kernel='rbf', gamma=0.001),
    SVC(kernel='rbf', gamma=1),
    SVC(kernel='rbf', gamma=20)]
names = ["SVM (rbf, $\gamma = 0.001$)", "SVM (rbf, $\gamma = 1$)", "SVM (rbf, $\gamma = 20$)"]
from matplotlib.colors import ListedColormap

figure = plt.figure(figsize=(9, 4))
i = 1
h = .02  # step size in the mesh
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

# iterate over classifiers
for name, clf in zip(names, classifiers):
    ax = plt.subplot(1, len(classifiers), i)
    clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(name)
    i += 1
plt.show()