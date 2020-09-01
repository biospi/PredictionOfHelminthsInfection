import numpy as np
import numba as nb
import time
from numba import jit
import math
import pandas as pd
from random import randrange


def isNaN(num):
    return num != num


def process(activity_list, thresh=10):
    i = 0
    cpt = 0
    row_to_del = ()
    to_del = ()
    gap = ()
    activity_list = tuple(activity_list)
    print("activity_list_size=", len(activity_list))
    for activity in activity_list:
        if cpt >= thresh:
            cpt = 0
            to_del = to_del + row_to_del
            row_to_del = ()

        if isNaN(activity):
            cpt += 1
            row_to_del = row_to_del + (i,)

            gap = gap + (i,)
            i += 1
            continue
        cpt = 0
        row_to_del = ()
        gap = ()
        i += 1
    to_del = list(to_del)
    #remove chunck in df ....


def dummy_activity():
    x = [np.nan] * 666323
    rnd_indexes = [randrange(len(x)) for _ in range(int(len(x)/4))]
    x = np.array(x)
    x[rnd_indexes] = 10
    return x.tolist()


if __name__ == "__main__":
    print("start...")
    start = time.time()
    process(dummy_activity())
    end = time.time()
    print("Elapsed (after compilation) = %s" % (end - start))
