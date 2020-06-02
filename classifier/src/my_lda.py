import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.model_selection import train_test_split
from sys import exit
import scipy

style.use('fivethirtyeight')
from sklearn.neighbors import KNeighborsClassifier


def process_lda(X, y, n_features):
    # 0. Load in the data and split the descriptive and the target feature
    # df = pd.read_csv('data/Wine.txt', sep=',',
    #                  names=['target', 'Alcohol', 'Malic_acid', 'Ash', 'Akcakinity', 'Magnesium', 'Total_pheonols',
    #                         'Flavanoids', 'Nonflavanoids', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280',
    #                         'Proline'])
    # X = df.iloc[:, 1:].copy()
    # target = df['target'].copy()
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.3, random_state=0)
    #
    # # 1. Standardize the data
    # for col in X_train.columns:
    #     X_train[col] = StandardScaler().fit_transform(X_train[col].values.reshape(-1, 1))

    # 2. Compute the mean vector mu and the mean vector per class mu_k
    print("Compute the mean vector mu and the mean vector per class mu_k")
    X = pd.DataFrame(X)
    y = pd.Series(y)
    df = X.copy()
    df['target'] = y
    for col in X.columns:
        X[col] = StandardScaler().fit_transform(X[col].values.reshape(-1, 1))
    mu = np.mean(X, axis=0).values.reshape(X.shape[1], 1)  # Mean vector mu --> Since the data has been standardized, the data means are zero

    mu_k = []

    for i, orchid in enumerate(np.unique(df['target'])):
        mu_k.append(np.mean(X.where(df['target'] == orchid), axis=0))
    mu_k = np.array(mu_k).T

    # 3. Compute the Scatter within and Scatter between matrices
    print("Compute the Scatter within and Scatter between matrices")
    data_SW = []
    Nc = []
    for i, orchid in enumerate(np.unique(df['target'])):
        a = np.array(X.where(df['target'] == orchid).dropna().values - mu_k[:, i].reshape(1, X.shape[1]))
        data_SW.append(np.dot(a.T, a))
        Nc.append(np.sum(df['target'] == orchid))
    SW = np.sum(data_SW, axis=0)

    SB = np.dot(Nc * np.array(mu_k - mu), np.array(mu_k - mu).T)

    # 4. Compute the Eigenvalues and Eigenvectors of SW^-1 SB
    print("Compute the Eigenvalues and Eigenvectors of")
    eigval, eigvec = np.linalg.eig(np.dot(np.linalg.inv(SW), SB))

    # 5. Select the two largest eigenvalues
    print("5")
    eigen_pairs = [[np.abs(eigval[i]), eigvec[:, i]] for i in range(len(eigval))]
    eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
    w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real, eigen_pairs[1][1][:, np.newaxis].real))
    print(w)

    # 6. Transform the data with Y=X*w
    print("6")
    # 6. Transform the data with Y=X*w
    Y = X.dot(w)

    # Plot the data
    fig = plt.figure(figsize=(10, 10))
    ax0 = fig.add_subplot(111)
    ax0.set_xlim(-0.5, 0.5)
    ax0.set_ylim(-0.5, 0.5)

    un = np.unique(y)
    for l, c, m in zip(un, ['r', 'g', 'b'][0:len(un)], ['s', 'x', 'o'][0:len(un)]):
        ax0.scatter(Y[0][y == l],
                    Y[1][y == l],
                    c=c, marker=m, label=l, edgecolors='black')
    ax0.legend(loc='upper right')

    # Plot the voroni spaces
    print("7")

    for target in zip(np.unique(y)):
        ax0.scatter(np.mean(Y[y == target], axis=0)[0], np.mean(Y[y == target], axis=0)[1],
                    c='black',
                    s=100)

    # mesh_x, mesh_y = np.meshgrid(np.linspace(-3, 3), np.linspace(-4, 3))
    # mesh = []
    #
    # for i in range(len(mesh_x)):
    #     for j in range(len(mesh_x[0])):
    #         date = [mesh_x[i][j], mesh_y[i][j]]
    #         mesh.append((mesh_x[i][j], mesh_y[i][j]))

    # NN = KNeighborsClassifier(n_neighbors=1)
    # NN.fit(means, ['r', 'g', 'b'])
    # predictions = NN.predict(np.array(mesh))
    #
    # ax0.scatter(np.array(mesh)[:, 0], np.array(mesh)[:, 1], color=predictions, alpha=0.3)

    plt.show()
    exit()


if __name__ == '__main__':
    process_lda()