import matplotlib.pyplot as plt
from sklearn import datasets

X, y = datasets.make_blobs(n_samples=30, centers=2, n_features=2, center_box=(0, 10))


x1 = X[:, 0][y == 0]
y1 = X[:, 1][y == 0]
x2 = X[:, 0][y == 1]
y2 = X[:, 1][y == 1]
print(f"min={X.min():.1f} max={X.max():.1f}")
for i in range(x1.shape[0]):
    print(f"{x1[i]:.1f}, {y1[i]:.1f}, {x2[i]:.1f}, {y2[i]:.1f}")

plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'g^')
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs')
plt.show()