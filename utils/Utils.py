import math
import os
import time

import numpy as np
import pandas as pd

"""
Utility class for static methods
"""

# def anscombe_r(value):
#     try:
#         return (value*value) / 2
#     except TypeError as e:
#         print(e)
#
# def anscombe(value):
#     try:
#         return 2 * math.sqrt(value + (3 / 8))
#     except TypeError as e:
#         print(e)

def anscombe(arr, sigma_sq=0, alpha=1):
    """
    Generalized Anscombe variance-stabilizing transformation
    References:
    [1] http://www.cs.tut.fi/~foi/invansc/
    [2] M. Makitalo and A. Foi, "Optimal inversion of the generalized
    Anscombe transformation for Poisson-Gaussian noise", IEEE Trans.
    Image Process, 2012
    [3] J.L. Starck, F. Murtagh, and A. Bijaoui, Image  Processing
    and Data Analysis, Cambridge University Press, Cambridge, 1998)
    :param arr: variance-stabilized signal
    :param sigma_sq: variance of the Gaussian noise component
    :param alpha: scaling factor of the Poisson noise component
    :return: variance-stabilized array
    """
    v = np.maximum((arr / alpha) + (3. / 8.) + sigma_sq / (alpha ** 2), 0)
    f = 2. * np.sqrt(v)
    return f


def inverse_anscombe(arr, sigma_sq=0, m=0, alpha=1, method='closed-form'):
    """
    Inverse of the Generalized Anscombe variance-stabilizing
    transformation
    References:
    [1] http://www.cs.tut.fi/~foi/invansc/
    [2] M. Makitalo and A. Foi, "Optimal inversion of the generalized
    Anscombe transformation for Poisson-Gaussian noise", IEEE Trans.
    Image Process, 2012
    [3] J.L. Starck, F. Murtagh, and A. Bijaoui, Image  Processing
    and Data Analysis, Cambridge University Press, Cambridge, 1998)


    :param arr: variance-stabilized signal
    :param sigma_sq: variance of the Gaussian noise component
    :param m: mean of the Gaussian noise component
    :param alpha: scaling factor of the Poisson noise component
    :param method: 'closed_form' applies the closed-form approximation
    of the exact unbiased inverse. 'asym' applies the asymptotic
    approximation of the exact unbiased inverse.
    :return: inverse variance-stabilized array
    """
    sigma_sq /= alpha ** 2

    if method == 'closed-form':
        # closed-form approximation of the exact unbiased inverse:
        arr_trunc = np.maximum(arr, 0.8)
        inverse = ((arr_trunc / 2.) ** 2 + 0.25 * np.sqrt(1.5) * arr_trunc ** -1 - (11. / 8.) * arr_trunc ** -2 +
                   (5. / 8.) * np.sqrt(1.5) * arr_trunc ** -3 - (1. / 8.) - sigma_sq)
    elif method == 'asym':
        # asymptotic approximation of the exact unbiased inverse:
        inverse = (arr / 2.) ** 2 - 1. / 8 - sigma_sq
        # inverse = np.maximum(0, inverse)
    else:
        raise NotImplementedError('Only supports the closed-form')

    if alpha != 1:
        inverse *= alpha

    if m != 0:
        inverse += m

    return inverse


def center_signal(y, avg):
    y_centered = y - avg
    return y_centered


def create_rec_dir(path):
    dir_path = ""
    sub_dirs = path.split("/")
    for sub_dir in sub_dirs[0:]:
        if "." in sub_dir:
            continue
        dir_path += sub_dir + "/"
        # print("sub_folder=", dir_path)
        if not os.path.exists(dir_path):
            print("mkdir", dir_path)
            os.makedirs(dir_path)


def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))
