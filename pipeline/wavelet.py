import numpy as np
import matplotlib.pyplot as plt


def wavelet(n, w0):
    y = (np.pi ** -0.25) * np.exp(1j*w0 * n) * np.exp(-((n)**2 ) / 2)
    return y


if __name__ == "__main__":
    w0 = 100
    N = 100
    y = []
    x = list(np.arange(-10, 10, 0.1))
    for n in x:
        y.append(wavelet(n, w0))

    plt.plot(x, y)
    plt.show()


