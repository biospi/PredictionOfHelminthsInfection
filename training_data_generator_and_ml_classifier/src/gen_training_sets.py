import gc
import glob
import json
import math
import os.path
import pathlib
import shutil
import statistics
import sys
import time
from datetime import datetime, timedelta
from multiprocessing import Pool
from sys import exit

import dateutil.relativedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycwt as wavelet

from ipython_genutils.py3compat import xrange
# from matplotlib.colors import LinearSegmentedColormap
# from matplotlib.lines import Line2D
# from mlxtend.plotting import plot_decision_regions
# from scipy import interp
# from sklearn import preprocessing
# from sklearn.base import clone
# from sklearn.cross_decomposition import PLSRegression
# from sklearn.decomposition import PCA
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import auc
# from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
# from sklearn.metrics import f1_score
# from sklearn.metrics import plot_roc_curve
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import normalize
# from sklearn.svm import SVC

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

import os

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 5)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

ts = time.time()
run_timestamp = datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
print(run_timestamp)

skipped_class_false, skipped_class_true = -1, -1

def purge_file(filename):
    print("purge %s..." % filename)
    try:
        os.remove(filename)
    except FileNotFoundError:
        print("file not found.")


def get_weight(curr_datetime, data_famacha_dict, animal_id):
    weight = None
    c = curr_datetime.strftime("%d/%m/%Y")
    for data in data_famacha_dict[str(animal_id)]:
        if c in data[0]:
            try:
                weight = float(data[3])
            except ValueError as e:
                # print(e)
                weight = None
    return weight


def get_temp_humidity(date, data):
    humidity = None
    temp = None
    try:
        data_d = data[date.strftime("%Y-%m-%d")]
        time_d = date.strftime("%H:%M:%S").split(':')[0]
        for item in data_d:
            if time_d == item['time'].split(':')[0]:
                humidity = int(item['humidity'])
                temp = int(item['temp_c'])
    except KeyError as e:
        # print("could not find weather data!", e)
        pass

    return temp, humidity


def get_prev_famacha_score(serial_number, famacha_test_date, data_famacha, curr_score):
    previous_score1 = None
    previous_score2 = None
    previous_score3 = None
    previous_score4 = None
    try:
        list = data_famacha[str(serial_number)]
    except KeyError as e:
        # print(e)
        exit()
    for i in range(1, len(list)):
        item = list[i]
        if item[0] == famacha_test_date:
            try:
                previous_score1 = int(list[i - 1][1])
            except ValueError as e:
                previous_score1 = -1
                # print(e)
            try:
                previous_score2 = int(list[i - 2][1])
            except ValueError as e:
                previous_score2 = -1
                # print(e)
            try:
                previous_score3 = int(list[i - 3][1])
            except ValueError as e:
                previous_score3 = -1
                # print(e)
            try:
                previous_score4 = int(list[i - 4][1])
            except ValueError as e:
                previous_score4 = -1
                # print(e)
            break
    return previous_score1, previous_score2, previous_score3, previous_score4


def pad(a, N):
    a += [-1] * (N - len(a))
    return a


def get_elapsed_time_string(time_initial, time_next):
    dt1 = datetime.fromtimestamp(time_initial)
    dt2 = datetime.fromtimestamp(time_next)
    rd = dateutil.relativedelta.relativedelta(dt2, dt1)
    return '%02d:%02d:%02d:%02d' % (rd.days, rd.hours, rd.minutes, rd.seconds)


def anscombe(value):
    return 2 * math.sqrt(abs(value) + (3 / 8))


def anscombe_list(activity):
    return [anscombe(x) if x is not None else None for x in activity]


def anscombe_log_list(activity):
    return [math.log(anscombe(x)) if x is not None else None for x in activity]


def normalize_histogram_mean_diff(activity_mean, activity):
    scale = [0 for _ in range(0, len(activity))]
    idx = []
    for n, a in enumerate(activity):
        if np.isnan(a) or a==0 or np.isnan(activity_mean[n]):
            continue
        r = activity_mean[n] / a
        scale[n] = r
        idx.append(n)
    median = math.fabs(statistics.median(sorted(set(scale))))
    for i in idx:
        activity[i] = activity[i] * median
    return activity


def get_period(curr_data_famacha, days_before_famacha_test, sliding_window=0):
    famacha_test_date = time.strptime(curr_data_famacha[0], "%d/%m/%Y")
    famacha_test_date_epoch_s = str(time.mktime((datetime.fromtimestamp(time.mktime(famacha_test_date)) -
                                                 timedelta(days=sliding_window)).timetuple())).split('.')[0]
    famacha_test_date_epoch_before_s = str(time.mktime((datetime.fromtimestamp(time.mktime(famacha_test_date)) -
                                                        timedelta(days=sliding_window + days_before_famacha_test)).
                                                       timetuple())).split('.')[0]
    famacha_test_date_formated = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(famacha_test_date_epoch_s)))
    famacha_test_date_epoch_before_formated = time.strftime('%Y-%m-%d %H:%M:%S',
                                                            time.localtime(int(famacha_test_date_epoch_before_s)))
    return famacha_test_date_epoch_s, famacha_test_date_epoch_before_s, famacha_test_date_formated, famacha_test_date_epoch_before_formated


def sql_dict_list_to_list(dict_list):
    return pd.DataFrame(dict_list)['serial_number'].tolist()


# def format_cedara_famacha_data(data_famacha_dict, sql_db):
#     # famacha records contains shorten version of the serial numbers whereas activity records contains full version
#     rows_serial_numbers = execute_sql_query(sql_db,
#                                             "SELECT DISTINCT serial_number FROM cedara_70091100056_resolution_10min")
#     serial_numbers_full = sql_dict_list_to_list(rows_serial_numbers)
#
#     data_famacha_dict_formatted = {}
#     for key, value in data_famacha_dict.items():
#         for elem in serial_numbers_full:
#             if key in str(elem):
#                 v = value.copy()
#                 for item in v:
#                     item[2] = elem
#                 data_famacha_dict_formatted[str(elem)] = v
#
#     return data_famacha_dict_formatted


def get_ndays_between_dates(date1, date2):
    date_format = "%d/%m/%Y"
    a = datetime.strptime(date1, date_format)
    b = datetime.strptime(date2, date_format)
    delta = b - a
    return delta.days


def get_training_data(csv_df, csv_median_df, curr_data_famacha, i, data_famacha_list, data_famacha_dict, weather_data, resolution,
                      days_before_famacha_test):
    # print("generating new training pair....")
    famacha_test_date = datetime.fromtimestamp(time.mktime(time.strptime(curr_data_famacha[0], "%d/%m/%Y"))).strftime(
        "%d/%m/%Y")
    try:
        famacha_score = int(curr_data_famacha[1])
    except ValueError as e:
        # print("error while parsing famacha score!", e)
        return

    animal_id = int(curr_data_famacha[2])
    # find the activity data of that animal the n days before the test
    date1, date2, _, _ = get_period(curr_data_famacha, days_before_famacha_test)

    dtf1, dtf2, dtf3, dtf4, dtf5 = "", "", "", "", ""
    try:
        dtf1 = data_famacha_list[i][0]
    except IndexError as e:
        # print(e)
        pass
    try:
        dtf2 = data_famacha_list[i + 1][0]
    except IndexError as e:
        # print(e)
        pass
    try:
        dtf3 = data_famacha_list[i + 2][0]
    except IndexError as e:
        # print(e)
        pass
    try:
        dtf4 = data_famacha_list[i + 3][0]
    except IndexError as e:
        # print(e)
        pass
    try:
        dtf5 = data_famacha_list[i + 4][0]
    except IndexError as e:
        # print(e)
        pass

    nd1, nd2, nd3, nd4 = 0, 0, 0, 0
    if len(dtf2) > 0 and len(dtf1) > 0:
        nd1 = abs(get_ndays_between_dates(dtf1, dtf2))
    if len(dtf3) > 0 and len(dtf2) > 0:
        nd2 = abs(get_ndays_between_dates(dtf2, dtf3))
    if len(dtf4) > 0 and len(dtf3) > 0:
        nd3 = abs(get_ndays_between_dates(dtf3, dtf4))
    if len(dtf5) > 0 and len(dtf4) > 0:
        nd4 = abs(get_ndays_between_dates(dtf4, dtf5))

    # print("getting activity data for test on the %s for %d. collecting data %d days before resolution is %s..." % (famacha_test_date, animal_id, days_before_famacha_test, resolution))

    rows_activity, time_range = execute_df_query(csv_df, animal_id, resolution, date2, date1)
    rows_herd, _ = execute_df_query(csv_median_df, "median animal", resolution, date2, date1)

    herd_activity_list = rows_herd
    herd_activity_list_raw = herd_activity_list.copy()
    activity_list = rows_activity
    activity_list_raw = activity_list.copy()

    expected_sample_count = get_expected_sample_count("1min", days_before_famacha_test)

    if len(rows_activity) < expected_sample_count:
        l = len(rows_activity)
        # print("absent activity records. skip.", "found %d" % l, "expected %d" % expected_sample_count)
        return

    # activity_list = normalize_histogram_mean_diff(herd_activity_list, activity_list)
    # herd_activity_list = anscombe_list(herd_activity_list)
    # activity_list = anscombe_list(activity_list)

    idx = 0
    indexes = []
    timestamp_list = []
    humidity_list = []
    weight_list = []
    temperature_list = []
    dates_list_formated = []

    for j, data_a in enumerate(rows_activity[0: expected_sample_count]):
        # transform date in time for comparaison
        curr_datetime = time_range[j]
        timestamp = time.strptime(curr_datetime.strftime('%d/%m/%Y'), "%d/%m/%Y")
        # temp, humidity = get_temp_humidity(curr_datetime, weather_data)
        temp, humidity = 0, 0

        weight = None
        try:
            weight = float(curr_data_famacha[3])
        except ValueError as e:
            # print("weight=", weight)
            # print(e)
            pass

        weight_list.append(weight)

        indexes.append(idx)
        timestamp_list.append(timestamp)
        temperature_list.append(temp)
        humidity_list.append(humidity)
        dates_list_formated.append(curr_datetime.strftime('%d/%m/%Y %H:%M'))
        idx += 1

    prev_famacha_score1, prev_famacha_score2, prev_famacha_score3, prev_famacha_score4 = get_prev_famacha_score(
        animal_id,
        famacha_test_date,
        data_famacha_dict,
        famacha_score)
    indexes.reverse()

    data = {"famacha_score_increase": False, "famacha_score": famacha_score, "weight": weight_list,
            "previous_famacha_score1": prev_famacha_score1,
            "previous_famacha_score2": prev_famacha_score2,
            "previous_famacha_score3": prev_famacha_score3,
            "previous_famacha_score4": prev_famacha_score4,
            "time_range": time_range,
            "animal_id": animal_id,
            "date_range": [time.strftime('%d/%m/%Y', time.localtime(int(date1))),
                           time.strftime('%d/%m/%Y', time.localtime(int(date2)))],
            "dtf1": dtf1,
            "dtf2": dtf2,
            "dtf3": dtf3,
            "dtf4": dtf4,
            "dtf5": dtf5,
            "nd1": nd1,
            "nd2": nd2,
            "nd3": nd3,
            "nd4": nd4,
            "indexes": indexes, "activity": activity_list, "activity_raw": activity_list_raw,
            "temperature": temperature_list, "humidity": humidity_list, "herd": herd_activity_list,
            "herd_raw": herd_activity_list_raw, "ignore": True
            }
    return data


def get_futures_results(futures):
    print("getting results from future...")
    results = []
    for future in futures:
        res = future.result()
        if res is None:
            continue
        results.append(res)
    return results


def save_result_in_file(results, filename):
    print("saving %d results in file..." % len(results))
    with open(filename, 'a') as outfile:
        for res in results:
            json.dump(res, outfile)
            outfile.write('\n')


def process_famacha_var(results, count_skipped=True):
    print("computing classes...")
    skipped_class_false, skipped_class_true = 0, 0
    for i in xrange(0, len(results) - 1):
        curr_data = results[i]
        next_data = results[i + 1]
        if curr_data["animal_id"] != next_data["animal_id"]:  # not same animal id
            if count_skipped:
                skipped_class_false += 1
            continue
        if curr_data["famacha_score"] == next_data["famacha_score"]:  # same famacha score
            next_data["famacha_score_increase"] = False
            next_data["ignore"] = False
        if curr_data["famacha_score"] < next_data["famacha_score"]:
            print("famacha score changed from 1 to >2. creating new set...")
            next_data["famacha_score_increase"] = True
            next_data["ignore"] = False
        if curr_data["famacha_score"] > next_data["famacha_score"]:
            print("famacha score changed decreased. creating new set...")
            next_data["famacha_score_increase"] = False
            next_data["ignore"] = False
    return skipped_class_false, skipped_class_true


def get_expected_sample_count(resolution, days_before_test):
    expected_sample_n = None
    if resolution == "1min":
        expected_sample_n = (24 * 60) * days_before_test
    if resolution == "5min":
        expected_sample_n = ((24 * 60) / 5) * days_before_test
    if resolution == "10min":
        expected_sample_n = ((24 * 60) / 10) * days_before_test
    if resolution == "hour":
        expected_sample_n = (24 * days_before_test)
    if resolution == "day":
        expected_sample_n = days_before_test

    expected_sample_n = expected_sample_n + 1 #todo fix clipping
    # print("expected sample count is %d." % expected_sample_n)
    return int(expected_sample_n)


def is_activity_data_valid(activity_raw):
    if np.isnan(activity_raw).any():
        return False, 'has_nan'
    return True, 'ok'


def even_list(n):
    result = [1]
    for num in range(2, n * 2 + 1, 2):
        result.append(num)
    del result[-1]
    return np.asarray(result, dtype=np.int32)


def create_activity_graph(animal_id, activity, folder, filename, title=None,
                          sub_folder='training_sets_time_domain_graphs', sub_sub_folder=None):
    activity = [-10 if x == 0 else x for x in activity] #set 0 value to 10 for ease of visualisation
    # for a in activity:
    #     if np.isnan(a) or a == None:
    #         raise ValueError("There should not be nan in trace at this stage!")

    fig = plt.figure()
    plt.bar(range(0, len(activity)), activity)
    fig.suptitle(title, x=0.5, y=.95, horizontalalignment='center', verticalalignment='top', fontsize=10)
    path = "%s/%s/%s" % (folder, sub_folder, sub_sub_folder)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    fig.savefig('%s/%s_%d_%s' % (path, animal_id, len(activity), filename.replace(".", "_")))
    # print(len(activity), filename.replace(".", "_"))
    fig.clear()
    plt.close(fig)


def create_hd_cwt_graph(coefs, cwt_lengh, folder, filename, title=None, sub_folder='training_sets_cwt_graphs',
                        sub_sub_folder=None, freqs=None):
    fig, axs = plt.subplots(1)
    # ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    # ax.imshow(coefs)
    # ax.set_yscale('log')
    t = [v for v in range(coefs.shape[1])]
    axs.pcolor(t, freqs, coefs)
    axs.set_ylabel('Frequency')
    axs.set_yscale('log')

    path = "%s/%s/%s" % (folder, sub_folder, sub_sub_folder)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    fig.savefig('%s/%d_%s' % (path, cwt_lengh, filename.replace('.', '')))
    # print(cwt_lengh, filename.replace('.', ''))
    fig.clear()
    plt.close(fig)

    # fig.savefig('%s/%s' % (path, filename))


def mask_cwt(cwt, coi):
    print("masking cwt...")
    for i in range(coi.shape[0]):
        col = cwt[:, i]
        max_index = int(coi[i])
        indexes_to_keep = np.array(list(range(max_index, col.shape[0])))
        total_indexes = np.array(range(col.shape[0]))
        diff = list(set(indexes_to_keep).symmetric_difference(total_indexes))
        # print(indexes_to_keep)
        if len(indexes_to_keep) == 0:
            continue
        col[indexes_to_keep] = -1
    return cwt


def compute_cwt(activity, scale=80):
    # print("compute_cwt...")
    # t, activity = dummy_sin()
    scales = even_list(scale)
    num_steps = len(activity)
    x = np.arange(num_steps)
    y = activity

    delta_t = (x[1] - x[0]) * 1
    # scales = np.arange(1, int(num_steps/10))
    freqs = 1 / (wavelet.MexicanHat().flambda() * scales)
    wavelet_type = 'mexicanhat'
    # y = [0 if x is np.nan else x for x in y] #todo fix
    coefs, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(y, delta_t, wavelet=wavelet_type, freqs=freqs)

    # coefs_masked = mask_cwt(coefs.real, coi)
    coefs_masked = coefs.real
    cwt = []
    for element in coefs_masked:
        for tupl in element:
            # if np.isnan(tupl) or tupl < 0:
            #     continue
            cwt.append(tupl)
    # indexes = np.asarray(list(range(len(coefs.real))))
    indexes = []
    return cwt, coefs.real, freqs, indexes, scales, delta_t, wavelet_type, coi



def create_filename(data):
    filename = "famacha[%.1f]_%d_%s_%s_sd_pvfs%d.png" % (data["famacha_score"], data["animal_id"], data["date_range"][0], data["date_range"][1],
    -1 if data["previous_famacha_score1"] is None else data["previous_famacha_score1"])
    return filename.replace('/', '-').replace(".", "_")


def create_training_set(result, dir, resolution, days_before_famacha_test, farm_id, options=[]):
    training_set = []
    option = ""
    if "cwt" in options:
        training_set.extend(result["coef_shape"])
        training_set.extend(result["cwt"])
        option = option + "cwt_"
    if "cwt_weight" in options:
        training_set.extend(result["cwt_weight"])
        option = option + "cwt_weight_"
    if "indexes_cwt" in options:
        training_set.extend(result["indexes_cwt"])
        option = option + "indexes_cwt_"
    if "activity" in options:
        training_set.extend(result["activity"])
        option = option + "activity_"
    if "indexes" in options:
        training_set.extend(result["indexes"])
        option = option + "indexes_"
    if "weight" in options:
        training_set.extend(result["weight"])
        option = option + "weight_"
    if "humidity" in options:
        training_set.extend(result["humidity"])
        option = option + "humidity_"
    if "temperature" in options:
        training_set.extend(result["temperature"])
        option = option + "temperature_"
    if "famacha_score" in options:
        training_set.append(result["famacha_score"])
        option = option + "famacha_score_"
    if "previous_score" in options:
        training_set.append(result["previous_famacha_score1"])
        option = option + "previous_score_"
    training_set.append(result["famacha_score_increase"])
    training_set.append(len(training_set))
    training_set.extend(result["date_range"])
    training_set.append(result["animal_id"])
    training_set.append(result["famacha_score"])
    training_set.append(result["previous_famacha_score1"])
    training_set.append(result["previous_famacha_score2"])
    training_set.append(result["previous_famacha_score3"])
    training_set.append(result["previous_famacha_score4"])
    training_set.append(result["dtf1"])
    training_set.append(result["dtf2"])
    training_set.append(result["dtf3"])
    training_set.append(result["dtf4"])
    training_set.append(result["dtf5"])
    training_set.append(result["nd1"])
    training_set.append(result["nd2"])
    training_set.append(result["nd3"])
    training_set.append(result["nd4"])
    path = "%s/training_sets" % dir
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    filename = "%s/%s_%s_dbft%d_%s.data" % (path, option, farm_id, days_before_famacha_test, resolution)
    training_str_flatten = str(training_set).strip('[]').replace(' ', '').replace('None', 'NaN')
    # print("set size is %d, %s.....%s" % (
    #     len(training_set), training_str_flatten[0:50], training_str_flatten[-50:]))
    with open(filename, 'a') as outfile:
        outfile.write(training_str_flatten)
        outfile.write('\n')

    # filename_t = "%s/temperature.data" % path
    # temp_str_flatten = str(result["temperature"]).strip('[]').replace(' ', '').replace('None', 'NaN')
    # print("set size is %d, %s.....%s" % (
    #     len(temp_str_flatten), temp_str_flatten[0:50], temp_str_flatten[-50:]))
    # with open(filename_t, 'a') as outfile_t:
    #     outfile_t.write(temp_str_flatten)
    #     outfile_t.write('\n')
    #
    # filename_h = "%s/humidity.data" % path
    # hum_str_flatten = str(result["humidity"]).strip('[]').replace(' ', '').replace('None', 'NaN')
    # print("set size is %d, %s.....%s" % (
    #     len(hum_str_flatten), hum_str_flatten[0:50], hum_str_flatten[-50:]))
    # with open(filename_h, 'a') as outfile_h:
    #     outfile_h.write(hum_str_flatten)
    #     outfile_h.write('\n')

    return filename, options


def create_training_sets(data, dir_path, resolution, days_before_famacha_test, farm_id):
    path1, options1 = create_training_set(data, dir_path, resolution, days_before_famacha_test, farm_id, options=["activity"])
    path2, options2 = create_training_set(data, dir_path, resolution, days_before_famacha_test, farm_id, options=["cwt"])

    return [
        {"path": path1, "options": options1},
        {"path": path2, "options": options2}
    ]


def create_graph_title(data, domain):
    hum_1 = ','.join([str(int(x)) for x in data["humidity"][0:1]])
    hum_2 = ','.join([str(int(x)) for x in data["humidity"][-1:]])
    temp_1 = ','.join([str(int(x)) for x in data["temperature"][0:1]])
    temp_2 = ','.join([str(int(x)) for x in data["temperature"][-1:]])
    if domain == "time":
        act_1 = ','.join([str(int(x)) for x in data["activity"][0:1] if x is not None])
        act_2 = ','.join([str(int(x)) for x in data["activity"][-1:] if x is not None])
        idxs_1 = ','.join([str(int(x)) for x in data["indexes"][0:1]])
        idxs_2 = ','.join([str(int(x)) for x in data["indexes"][-1:]])
        return "[[%s...%s],[%s...%s],[%s...%s],[%s...%s],%d,%d,%s]" % (
            act_1, act_2, idxs_1, idxs_2, hum_1, hum_2,
            temp_1, temp_2, data["famacha_score"],
            -1 if data["previous_famacha_score1"] is None else data["previous_famacha_score1"],
            str(data["famacha_score_increase"]))
    if domain == "freq":
        idxs_1 = ','.join([str(int(x)) for x in data["indexes_cwt"][0:1]])
        idxs_2 = ','.join([str(int(x)) for x in data["indexes_cwt"][-1:]])
        cwt_1 = ','.join([str(int(x)) for x in data["cwt"][0:1]])
        cwt_2 = ','.join([str(int(x)) for x in data["cwt"][-1:]])
        return "[cwt:[%s...%s],idxs:[%s...%s],h:[%s...%s],t:[%s...%s],fs:%d,pfs:%d,%s]" % (
            cwt_1, cwt_2, idxs_1, idxs_2, hum_1, hum_2,
            temp_1, temp_2, data["famacha_score"],
            -1 if data["previous_famacha_score1"] is None else data["previous_famacha_score1"],
            str(data["famacha_score_increase"]))


def load_db_from_csv(csv_db_path):
    print("loading data from csv file...")
    df = pd.read_csv(csv_db_path, sep=",")
    return df


def get_famacha_data(famacha_data_file_path):
    with open(famacha_data_file_path, 'r') as fp:
        data_famacha_dict = json.load(fp)
        print("FAMACHA=", data_famacha_dict.keys())
        # if 'cedara' in farm_id:
        #     data_famacha_dict = format_cedara_famacha_data(data_famacha_dict, None) #todo fix
    return data_famacha_dict


def execute_df_query(csv_df, animal_id, resolution, date2, date1):
    print("execute df query...", animal_id, resolution, date2, date1)
    df = csv_df.copy()
    df["datetime64"] = pd.to_datetime(df['date_str'])
    start = pd.to_datetime(datetime.fromtimestamp(int(date2)))
    end = pd.to_datetime(datetime.fromtimestamp(int(date1)))
    df_ = df[df.datetime64.between(start, end)]
    activity = df_["first_sensor_value"].to_list()
    time_range = df_["timestamp"].to_list()
    try:
        time_range = [datetime.fromtimestamp(x) for x in time_range]
    except ValueError as e:
        pass
        # print("error while converting time", e)
        # print(time_range)
        # print(csv_df)

    return activity, time_range


def resample_traces(resolution, activity, herd):
    index = pd.date_range('1/1/2000', periods=len(activity), freq='T')
    activity_series = pd.Series(activity, index=index)
    activity_series_r = activity_series.resample(resolution).sum()
    activity_r = activity_series_r.values
    herd_series = pd.Series(herd, index=index)
    herd_series_r = herd_series.resample(resolution).sum()
    herd_r = herd_series_r.values
    return activity_r, herd_r


def process_day(thresh_i, thresh_z2n, days_before_famacha_test, resolution, farm_id, csv_folder, file, file_median, data_famacha_dict, create_input_visualisation_eanable=False):
    # print("animal file=", file)
    csv_df = load_db_from_csv(file)
    csv_median = load_db_from_csv(file_median)
    dir = "%s/%s_%s_famachadays_%d_threshold_interpol_%d_threshold_zero2nan_%d" % (csv_folder, farm_id, resolution, days_before_famacha_test, thresh_i, thresh_z2n)
    create_cwt_graph_enabled = False
    create_activity_graph_enabled = True
    weather_data = None

    try:
        shutil.rmtree(dir, ignore_errors=True)
    except (OSError, FileNotFoundError) as e:
        # print(e)
        pass

    dataset_heatmap_data = {}
    data_famacha_list = [y for x in data_famacha_dict.values() for y in x]
    results = []
    for i, curr_data_famacha in enumerate(data_famacha_list):
        try:
            result = get_training_data(csv_df, csv_median, curr_data_famacha, i, data_famacha_list.copy(),
                                       data_famacha_dict, weather_data, resolution,
                                       days_before_famacha_test)
        except KeyError as e:
            result = None
            # print(e)

        if result is None:
            continue

        activity_resampled, herd_resampled = resample_traces(resolution, result["activity"], result["herd"])

        is_valid, reason = is_activity_data_valid(result["activity_raw"])
        # print("sample is valid=", is_valid)

        result["activity"] = activity_resampled.tolist()

        result['is_valid'] = is_valid
        if result["famacha_score"] < 0 or result["previous_famacha_score1"] < 0:
            result['is_valid'] = False
            reason = 'missing_f'

        result['reason'] = reason

        results.append(result)

        animal_id = str(curr_data_famacha[2])

        if animal_id not in dataset_heatmap_data.keys():
            dataset_heatmap_data[animal_id] = {"id": animal_id, "activity": [], "date": [], "famacha": [],
                                               "famacha_previous": [], "valid": []}

        activity = [np.nan if x is None else x for x in result["activity"]]
        dataset_heatmap_data[animal_id]["activity"].append(activity)
        dataset_heatmap_data[animal_id]["date"].append(result["time_range"])
        dataset_heatmap_data[animal_id]["famacha"].append(result["famacha_score"])
        dataset_heatmap_data[animal_id]["famacha_previous"].append(result["previous_famacha_score1"])
        dataset_heatmap_data[animal_id]["valid"].append(result["is_valid"])

        # if len(results) > 10:
        #     break

    skipped_class_false, skipped_class_true = process_famacha_var(results)
    # if create_input_visualisation_eanable:
    #     try:
    #         for i in range(len(dataset_heatmap_data.keys())):
    #             print(list(dataset_heatmap_data.values())[i]['famacha'])
    #         # create_herd_map(farm_id, meta_data, activity_data, animals_id, time_range, fontsize=50)
    #         f_id = farm_id + '_' + resolution + '_' + str(days_before_famacha_test) + "_nan" + str(thresh_i)
    #         create_dataset_map(dataset_heatmap_data, dir + "/" + f_id, chunck_size=len(activity))
    #         create_histogram(dataset_heatmap_data, dir + "/" +"rhistogram_"+f_id)
    #     except ValueError as e:
    #         print("error while creating input visualisation", e)
    #         print(dataset_heatmap_data)

    class_input_dict = []
    # print("create_activity_graph...")
    # print("could find %d samples." % len(results))
    for idx in range(len(results)):
        result = results[idx]
        sub_sub_folder = str(result["is_valid"]) + "/"
        pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
        filename_graph = create_filename(result)
        if create_activity_graph_enabled:

            create_activity_graph(str(result["famacha_score_increase"]) + "_" + str(result["animal_id"]), result["activity"], dir, filename_graph,
                                  title=create_graph_title(result, "time"),
                                  sub_sub_folder=sub_sub_folder)

        if not result['is_valid']:
            continue

        # print("result valid %d/%dfor %s." % (idx, len(results), str(result["animal_id"])))
        cwt, coef, freqs, indexes_cwt, scales, delta_t, wavelet_type, coi = compute_cwt(result["activity"])

        result["cwt"] = cwt
        result["coef_shape"] = coef.shape
        # result["cwt_weight"] = cwt_weight
        result["indexes_cwt"] = indexes_cwt

        if create_cwt_graph_enabled:
            create_hd_cwt_graph(coef, len(cwt), dir, filename_graph, title=create_graph_title(result, "freq"),
                                sub_sub_folder=sub_sub_folder, freqs=freqs)

        create_training_sets(result, dir, resolution, days_before_famacha_test, farm_id)
        results[idx] = None
        gc.collect()

        # print("create_activity_graph done.")


def parse_csv_db_name(path):
    split = path.split('/')[-3].split('_')
    farm_id = split[0] + "_" + split[1]
    threshold_interpol = int(path.split('/')[-2].split('_')[-3])
    threshold_zeros2nan = int(path.split('/')[-2].split('_')[-1])
    return farm_id, threshold_interpol, threshold_zeros2nan


if __name__ == '__main__':

    print("args: csv_db_dir_path famacha_file_path n_days_before_famacha resampling_resolution n_process")

    if len(sys.argv) > 1:
        csv_db_dir_path = sys.argv[1]
        famacha_file_path = sys.argv[2]
        n_days_before_famacha = int(sys.argv[3])
        resampling_resolution = sys.argv[4]
        n_process = int(sys.argv[5])
    else:
        exit(-1)

    print("csv_db_path=", csv_db_dir_path)
    print("famacha_file_path=", famacha_file_path)
    print("n_days_before_famacha=", n_days_before_famacha)
    print("resampling_resolution=", resampling_resolution)

    files = glob.glob(csv_db_dir_path+"*.csv")
    print("found %d files." % len(files))
    if len(files) == 0:
        print("no files in %s" % csv_db_dir_path)
        exit(-1)

    file_median = None
    for idx, file in enumerate(files):
        if 'median' in file:
            file_median = file
            break

    MULTI_THREADING_ENABLED = (n_process > 0)

    if MULTI_THREADING_ENABLED:
        pool = Pool(processes=n_process)
        for idx, file in enumerate(files):
            farm_id, thresh_i, thresh_z2n = parse_csv_db_name(csv_db_dir_path)
            pool.apply_async(process_day, (thresh_i, thresh_z2n, n_days_before_famacha, resampling_resolution, farm_id,
                        csv_db_dir_path.replace("/*.csv", ""), file, file_median,
                        get_famacha_data(famacha_file_path),))
        pool.close()
        pool.join()
    else:
        for idx, file in enumerate(files):
            farm_id, thresh_i, thresh_z2n = parse_csv_db_name(csv_db_dir_path)
            process_day(thresh_i, thresh_z2n, n_days_before_famacha, resampling_resolution, farm_id,
                        csv_db_dir_path.replace("/*.csv", ""), file, file_median,
                        get_famacha_data(famacha_file_path))
