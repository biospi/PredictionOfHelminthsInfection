import numpy as np
import numba as nb
import time
from numba import jit
import math
import pandas as pd
from random import randrange


def isNaN(num):
    return num != num


def using_clump(a):
    return [a[s] for s in np.ma.clump_unmasked(np.ma.masked_invalid(a))]


def process(activity_list, thresh=3):
    print("activity_list_size=", len(activity_list))
    df = pd.DataFrame(activity_list)
    df_ = df.copy()

    cpt = 0
    i = 0
    row_to_del = [np.nan] * len(activity_list)

    for activity in activity_list:
        if isNaN(activity):
            cpt += 1
            row_to_del[i] = i
        i += 1

    to_del_chunck = using_clump(row_to_del)

    for item in to_del_chunck:
        if len(item) > thresh:
            continue
        if item[0] == 0 or item[-1] == df.shape[0]:
            continue
        df[item[0]-1:item[-1] + 1 + 1] = df[item[0]-1:item[-1] + 1 + 1].interpolate()


    #remove chunck in df ....
    # df = df.drop(df.index[idx_to_interpolate])
    print(df.shape, df_.shape)


def dummy_activity():
    x = [np.nan] * 100
    rnd_indexes = [randrange(len(x)) for _ in range(int(len(x)/2))]
    x = np.array(x)
    for i in rnd_indexes:
        x[i] = randrange(20)
    return x.tolist()


def is_in_same_minute(date1, date2):
    if date1.date() != date2.date():
        return False
    if date1.hour != date2.hour:
        return False
    if date1.minute != date2.minute:
        return False
    return True


if __name__ == "__main__":
    print("start...")
    df = pd.DataFrame({"0": [1, 2, 3], "1": [4, 5, 6]})
    df.index.name = 'index'
    for row in df.itertuples():
        print(row.index, row.Index)
    res = is_in_same_minute(np.datetime64('2010-03-14T15:02:00.00').tolist(), np.datetime64('2010-03-14T15:02:56.00').tolist())
    print(res)
    start = time.time()
    process(dummy_activity())
    end = time.time()
    print("Elapsed (after compilation) = %s" % (end - start))
