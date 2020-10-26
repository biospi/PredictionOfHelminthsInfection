import matplotlib
from sys import platform as _platform
if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg')
import glob
import json
import math
import os.path
import pathlib
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
from sklearn.preprocessing import normalize

import pycwt as wavelet


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
    #try:
    #    os.remove(filename)
    #except FileNotFoundError:
    #    print("file not found.")


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
    try:
        famacha_data_list = data_famacha[str(serial_number)]
    except KeyError as e:
        print("error could not find famacha report data!", e)
        exit()
    for i, item in enumerate(famacha_data_list):
        previous_score1 = -1
        previous_score2 = -1
        previous_score3 = -1
        previous_score4 = -1

        if item[0] == famacha_test_date:
            idx1 = i - 1
            if idx1 >= 0:
                try:
                    previous_score1 = int(famacha_data_list[idx1][1])
                except ValueError:
                    pass
            idx2 = i - 2
            if idx2 >= 0:
                try:
                    previous_score2 = int(famacha_data_list[idx2][1])
                except ValueError:
                    pass
            idx3 = i - 3
            if idx3 >= 0:
                try:
                    previous_score3 = int(famacha_data_list[idx3][1])
                except ValueError:
                    pass
            idx4 = i - 4
            if idx4 >= 0:
                try:
                    previous_score4 = int(famacha_data_list[idx4][1])
                except ValueError:
                    pass
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


def get_previous_famacha_dates(data_famacha_list, i):
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
    return dtf1, dtf2, dtf3, dtf4, dtf5


def get_number_of_days_between_succesive_prev_test(dtf1, dtf2, dtf3, dtf4, dtf5):
    nd1, nd2, nd3, nd4 = 0, 0, 0, 0
    if len(dtf2) > 0 and len(dtf1) > 0:
        nd1 = abs(get_ndays_between_dates(dtf1, dtf2))
    if len(dtf3) > 0 and len(dtf2) > 0:
        nd2 = abs(get_ndays_between_dates(dtf2, dtf3))
    if len(dtf4) > 0 and len(dtf3) > 0:
        nd3 = abs(get_ndays_between_dates(dtf2, dtf4))
    if len(dtf5) > 0 and len(dtf4) > 0:
        nd4 = abs(get_ndays_between_dates(dtf4, dtf5))
    return nd1, nd2, nd3, nd4


def get_training_data(csv_df, csv_median_df, csv_mean_df, curr_data_famacha, i, data_famacha_list, data_famacha_dict, weather_data,
                      days_before_famacha_test):
    # print("generating new training pair....")
    # todo:sort out time
    # famacha_test_date = datetime.fromtimestamp(time.mktime(time.strptime(curr_data_famacha[0], "%d/%m/%Y"))).strftime(
    #     "%d/%m/%Y")
    famacha_test_date = curr_data_famacha[0]
    try:
        famacha_score = int(curr_data_famacha[1])
    except ValueError as e:
        # print("error while parsing famacha score!", e)
        return

    animal_id = int(curr_data_famacha[2])

    # # find the activity data of that animal the n days before the test
    date1, date2, _, _ = get_period(curr_data_famacha, days_before_famacha_test)

    dtf1, dtf2, dtf3, dtf4, dtf5 = get_previous_famacha_dates(data_famacha_list, i)

    nd1, nd2, nd3, nd4 = get_number_of_days_between_succesive_prev_test(dtf1, dtf2, dtf3, dtf4, dtf5)

    # print("getting activity data for test on the %s for %d. collecting data %d days before resolution is %s..." % (famacha_test_date, animal_id, days_before_famacha_test, resolution))

    rows_activity, time_range, nan_in_window = execute_df_query(csv_df, animal_id, date2, date1)
    #todo understand which mean or median
    median, _, _ = execute_df_query(csv_median_df, "median animal", date2, date1)
    mean, _, _ = execute_df_query(csv_mean_df, "mean animal", date2, date1)

    # rows_herd, _ = execute_df_query(csv_mean_df, "mean animal", resolution, date2, date1)
    mean_activity_list = mean
    median_activity_list = median
    activity_list = rows_activity
    activity_list_raw = activity_list


    expected_activity_point = get_expected_sample_count("1min", days_before_famacha_test)

    l = len(activity_list)
    if l < expected_activity_point:
        print("absent activity counts. skip.", "found %d" % l, "expected %d" % expected_activity_point)
        return

    idx = 0
    indexes = []
    timestamp_list = []
    humidity_list = []
    weight_list = []
    temperature_list = []
    dates_list_formated = []

    prev_famacha_score1, prev_famacha_score2, prev_famacha_score3, prev_famacha_score4 = get_prev_famacha_score(
        animal_id,
        famacha_test_date,
        data_famacha_dict,
        famacha_score)

    indexes.reverse()

    data = {"famacha_score_increase": False,
            "famacha_score": famacha_score,
            "weight": weight_list,
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
            "indexes": indexes,
            "activity": activity_list,
            "activity_raw": activity_list_raw,
            "temperature": temperature_list,
            "humidity": humidity_list,
            "median": median_activity_list,
            "mean": mean_activity_list,
            "ignore": True,
            "nan_in_window": nan_in_window
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
    for sample_data in results:
        famacha_score = sample_data['famacha_score']
        famacha_score_previous = sample_data['previous_famacha_score1']
        sample_data["ignore"] = True
        if famacha_score_previous == 1 and famacha_score == 1:
            sample_data["famacha_score_increase"] = False
            sample_data["ignore"] = False

        if famacha_score_previous == 1 and famacha_score >= 2:
            sample_data["famacha_score_increase"] = True
            sample_data["ignore"] = False

        # next_data = results[i + 1]
        # if curr_data["animal_id"] != next_data["animal_id"]:  # not same animal id
        #     if count_skipped:
        #         skipped_class_false += 1
        #     continue
        # if curr_data["famacha_score"] == next_data["famacha_score"]:  # same famacha score
        #     next_data["famacha_score_increase"] = False
        #     next_data["ignore"] = False
        # if curr_data["famacha_score"] < next_data["famacha_score"]:
        #     print("famacha score changed from 1 to >2. creating new set...")
        #     next_data["famacha_score_increase"] = True
        #     next_data["ignore"] = False
        # if curr_data["famacha_score"] > next_data["famacha_score"]:
        #     print("famacha score changed decreased. creating new set...")
        #     next_data["famacha_score_increase"] = False
        #     next_data["ignore"] = False
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


def create_rec_dir(path):
    dir_path = ""
    sub_dirs = path.split("/")
    for sub_dir in sub_dirs[0:]:
        dir_path += sub_dir+"/"
        # print("sub_folder=", dir_path)
        if not os.path.exists(dir_path):
            print("mkdir", dir_path)
            os.makedirs(dir_path)


def create_activity_graph(output_dir, animal_id, activity, folder, filename, title=None,
                          sub_folder='training_sets_time_domain_graphs', sub_sub_folder=None):
    # for a in activity:
    #     if np.isnan(a) or a == None:
    #         raise ValueError("There should not be nan in trace at this stage!")

    fig = plt.figure(figsize=(19.20, 10.80))
    plt.clf()
    plt.plot(activity)
    # plt.bar(range(0, len(activity)), activity)
    fig.suptitle(title, x=0.5, y=.95, horizontalalignment='center', verticalalignment='top', fontsize=10)

    path = "%s/%s/%s" % (output_dir, sub_folder, sub_sub_folder)
    create_rec_dir(path)
    filename = '%s/%s_%d_%s.png' % (path, animal_id, len(activity), filename.replace(".", "_"))
    fig.savefig(filename)
    # print(len(activity), filename.replace(".", "_"))
    fig.clear()
    plt.close(fig)


def create_hd_cwt_graph(output_dir, power_coefs, cwt_lengh, folder, filename, title=None, sub_folder='training_sets_cwt_graphs',
                        sub_sub_folder=None, freqs=None):
    fig, axs = plt.subplots(1)
    # ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    # ax.imshow(coefs)
    # ax.set_yscale('log')
    # t = [v for v in range(coefs.shape[0])]
    # axs.pcolor(t, freqs, coefs)
    # axs.set_ylabel('Frequency')
    # axs.set_yscale('log')
    plt.imshow(power_coefs, extent=[0, power_coefs.shape[1], 0, power_coefs.shape[0]], interpolation='nearest', aspect='auto')

    path = "%s/%s/%s" % (output_dir, sub_folder, sub_sub_folder)
    create_rec_dir(path)
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

# import pywt
# def compute_cwt(activity):
#     w = pywt.ContinuousWavelet('morl')
#     scales = even_list(40)
#     sampling_frequency = 1 / 60
#     sampling_period = 1 / sampling_frequency
#     coef, freqs = pywt.cwt(np.asarray(activity), scales, w, sampling_period=sampling_period)
#     cwt = [element for tupl in coef for element in tupl]
#     indexes = np.asarray(list(range(coef.shape[1])))
#     return cwt, coef, freqs, indexes, scales, 1, 'morlet'


def compute_cwt(activity, scale=120):
    # print("compute_cwt...")
    # t, activity = dummy_sin()
    scale = 300
    scales = even_list(scale)

    num_steps = len(activity)
    x = np.arange(num_steps)

    #
    delta_t = (x[1] - x[0]) * 1
    # # scales = np.arange(1, int(num_steps/10))
    freqs = 1 / (wavelet.Morlet().flambda() * scales)

    wavelet_type = 'morlet'
    # y = [0 if x is np.nan else x for x in y] #todo fix
    y = activity
    coefs, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(y, 1, wavelet=wavelet_type, freqs=freqs)
    coefs_cc = np.conj(coefs)
    power = np.real(np.multiply(coefs, coefs_cc))
    power_flatten = power.flatten()

    plt.imshow(power, extent=[0, power.shape[1], 0, power.shape[0]], interpolation='nearest',
               aspect='auto')
    plt.show()

    # coefs_masked = mask_cwt(coefs.real, coi)
    # coefs_masked = coefs.real
    # cwt = []
    # for element in coefs_masked:
    #     for tupl in element:
    #         # if np.isnan(tupl) or tupl < 0:
    #         #     continue
    #         cwt.append(tupl)
    # # indexes = np.asarray(list(range(len(coefs.real))))
    # indexes = []
    # print("cwt lengh=", len(cwt))
    return coefs.shape, power_flatten, power, freqs, [], scales, delta_t, wavelet_type, coi



def create_filename(data):
    filename = "famacha[%.1f]_%d_%s_%s_sd_pvfs%d.png" % (data["famacha_score"], data["animal_id"], data["date_range"][0], data["date_range"][1],
    -1 if data["previous_famacha_score1"] is None else data["previous_famacha_score1"])
    return filename.replace('/', '-').replace(".", "_")


def create_training_set(output_dir, result, dir, days_before_famacha_test, farm_id, thresh_i, thresh_z2n, options=[]):
    training_set = []
    training_set_median = []
    training_set_mean = []
    option = ""
    if "cwt" in options:
        training_set.extend(result["coef_shape"])
        training_set.extend(result["cwt"])
        option = option + "cwt"
    if "cwt_weight" in options:
        training_set.extend(result["cwt_weight"])
        option = option + "cwt_weight_"
    if "indexes_cwt" in options:
        training_set.extend(result["indexes_cwt"])
        option = option + "indexes_cwt_"

    if "activity" in options:
        training_set.extend(result["activity"])
        option = option + "activity"
        training_set_median.extend(result["median"])
        training_set_mean.extend(result["mean"])
        
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
    
    
    training_set_median.append("median"+"_"+str(result["famacha_score_increase"]))
    training_set_median.append(len(training_set_median))
    training_set_median.extend(result["date_range"])
    training_set_median.append(result["animal_id"])
    training_set_median.append(result["famacha_score"])
    training_set_median.append(result["previous_famacha_score1"])
    training_set_median.append(result["previous_famacha_score2"])
    training_set_median.append(result["previous_famacha_score3"])
    training_set_median.append(result["previous_famacha_score4"])
    training_set_median.append(result["dtf1"])
    training_set_median.append(result["dtf2"])
    training_set_median.append(result["dtf3"])
    training_set_median.append(result["dtf4"])
    training_set_median.append(result["dtf5"])
    training_set_median.append(result["nd1"])
    training_set_median.append(result["nd2"])
    training_set_median.append(result["nd3"])
    training_set_median.append(result["nd4"])


    training_set_mean.append("mean"+"_"+str(result["famacha_score_increase"]))
    training_set_mean.append(len(training_set_mean))
    training_set_mean.extend(result["date_range"])
    training_set_mean.append(result["animal_id"])
    training_set_mean.append(result["famacha_score"])
    training_set_mean.append(result["previous_famacha_score1"])
    training_set_mean.append(result["previous_famacha_score2"])
    training_set_mean.append(result["previous_famacha_score3"])
    training_set_mean.append(result["previous_famacha_score4"])
    training_set_mean.append(result["dtf1"])
    training_set_mean.append(result["dtf2"])
    training_set_mean.append(result["dtf3"])
    training_set_mean.append(result["dtf4"])
    training_set_mean.append(result["dtf5"])
    training_set_mean.append(result["nd1"])
    training_set_mean.append(result["nd2"])
    training_set_mean.append(result["nd3"])
    training_set_mean.append(result["nd4"])
    
    path = "%s/training_sets" % output_dir
    if not os.path.exists(path):
        print("mkdir", path)
        os.makedirs(path)
    filename = "%s/%s_%s_dbft_%d_%s_threshi_%d_threshz_%d.csv" % (path, option, farm_id, days_before_famacha_test, "1min", thresh_i, thresh_z2n)
    training_str_flatten = str(training_set).strip('[]').replace(' ', '').replace('None', 'NaN')

    training_str_median_flatten = str(training_set_median).strip('[]').replace(' ', '').replace('None', 'NaN')

    training_str_mean_flatten = str(training_set_mean).strip('[]').replace(' ', '').replace('None', 'NaN')

    # print("set size is %d, %s.....%s" % (
    #     len(training_set), training_str_flatten[0:50], training_str_flatten[-50:]))
    print("filename=", filename)
    with open(filename, 'a') as outfile:
        outfile.write(training_str_flatten+'\n')
        outfile.write(training_str_median_flatten + '\n')
        outfile.write(training_str_mean_flatten + '\n')
    outfile.close()

    print("appended dataset")
    return filename, options


def create_training_sets(output_dir, data, dir_path, days_before_famacha_test, farm_id, thresh_i, thresh_z2n):
    path1, options1 = create_training_set(output_dir, data, dir_path, days_before_famacha_test, farm_id, thresh_i, thresh_z2n, options=["activity"])
    # path2, options2 = create_training_set(output_dir, data, dir_path, resolution, days_before_famacha_test, farm_id,thresh_i, thresh_z2n, options=["cwt"])

    return [
        {"path": path1, "options": options1}
        # {"path": path2, "options": options2}
    ]


def create_graph_title(data, domain):
    hum_1 = ','.join([str(x) for x in data["humidity"][0:1]])
    hum_2 = ','.join([str(x) for x in data["humidity"][-1:]])
    temp_1 = ','.join([str(x) for x in data["temperature"][0:1]])
    temp_2 = ','.join([str(x) for x in data["temperature"][-1:]])
    if domain == "time":
        act_1 = ','.join([str(x) for x in data["activity"][0:3]])
        act_2 = ','.join([str(x) for x in data["activity"][-3:]])
        idxs_1 = ','.join([str(x) for x in data["indexes"][0:1]])
        idxs_2 = ','.join([str(x) for x in data["indexes"][-1:]])
        return "%d [[%s...%s],[%s...%s],[%s...%s],[%s...%s],%d,%s,%s]" % (int(data["nan_in_window"]),
            act_1, act_2, idxs_1, idxs_2, hum_1, hum_2,
            temp_1, temp_2, data["famacha_score"], str(data["previous_famacha_score1"]),
            str(data["famacha_score_increase"]))
    if domain == "freq":
        idxs_1 = ','.join([str(x) for x in data["indexes_cwt"][0:1]])
        idxs_2 = ','.join([str(x) for x in data["indexes_cwt"][-1:]])
        cwt_1 = ','.join([str(x) for x in data["cwt"][0:1]])
        cwt_2 = ','.join([str(x) for x in data["cwt"][-1:]])
        return "%d [cwt:[%s...%s],idxs:[%s...%s],h:[%s...%s],t:[%s...%s],fs:%d,pfs:%s,%s]" % (data["nan_in_window"],
            cwt_1, cwt_2, idxs_1, idxs_2, hum_1, hum_2,
            temp_1, temp_2, data["famacha_score"], str(data["previous_famacha_score1"]),
            str(data["famacha_score_increase"]))


def load_db_from_csv(csv_db_path, idx=-1):
    print("loading data from csv file %d" % (idx), csv_db_path)
    df = pd.read_csv(csv_db_path, sep=",")
    print("file %d loaded" % idx)
    return df


def get_famacha_data(famacha_data_file_path):
    with open(famacha_data_file_path, 'r') as fp:
        data_famacha_dict = json.load(fp)
        print("FAMACHA=", data_famacha_dict.keys())
        # if 'cedara' in farm_id:
        #     data_famacha_dict = format_cedara_famacha_data(data_famacha_dict, None) #todo fix
    return data_famacha_dict


def execute_df_query(csv_df, animal_id, date2, date1):
    # print("execute df query...", animal_id, resolution, date2, date1)
    if csv_df is None:
        print("could not load csv_df", animal_id)
        return [], [], []
    df = csv_df.copy()
    df["datetime64"] = pd.to_datetime(df['date_str'])
    start = pd.to_datetime(datetime.fromtimestamp(int(date2)))
    end = pd.to_datetime(datetime.fromtimestamp(int(date1)))

    df_ = df[df.datetime64.between(start, end)]

    nan_in_window = df_["first_sensor_value"].isna().sum()

    # if nan_in_window < 60*6:
    #     df_ = df_.fillna(-1)

    activity = df_["first_sensor_value"].to_list()

    time_range = df_["timestamp"].to_list()
    try:
        time_range = [datetime.fromtimestamp(x) for x in time_range]
    except ValueError as e:
        print("error while converting time", e)
        print(time_range)
        print(csv_df)

    return activity, time_range, nan_in_window


# def resample_traces(activity, herd):
#     if resolution == '1min':
#         return np.array(activity), np.array(herd)
#
#     index = pd.date_range('1/1/2000', periods=len(activity), freq='T')
#
#     activity_series = pd.Series(activity, index=index)
#     activity_series_r = activity_series.resample(resolution).sum()
#     activity_r = activity_series_r.values
#
#     if len(herd) > 0:
#         herd_series = pd.Series(herd, index=index)
#         herd_series_r = herd_series.resample(resolution).sum()
#         herd_r = herd_series_r.values
#     else:
#         herd_r = np.array([])
#
#     return activity_r, herd_r


def process_day(enable_graph_output, result_output_dir, csv_median, csv_mean, idx, thresh_i, thresh_z2n, days_before_famacha_test,
                farm_id, file, data_famacha_dict, create_input_visualisation_eanable=False):
    csv_df = load_db_from_csv(file, idx)
    # result_output_dir = "%s/%s_%s_famachadays_%d_threshold_interpol_%d_threshold_zero2nan_%d" % (output_dir, farm_id, resolution, days_before_famacha_test, thresh_i, thresh_z2n)
    create_cwt_graph_enabled = enable_graph_output
    create_activity_graph_enabled = enable_graph_output
    weather_data = None
    print("file=", file)
    print("split=", file.split('/'))
    try:
        animal_id = int(file.split('/')[-1].split('_')[0])
    except Exception as e:
        print("not a valid animal id", e, file)
        return

    print("result_output_dir=", result_output_dir)
    if not os.path.exists(result_output_dir):
        print("result_output_dir mkdir=")
        os.makedirs(result_output_dir)


    data_famacha_list = [y for x in data_famacha_dict.values() for y in x if y[2] == animal_id]

    results = []
    print("processing file %d" % idx)
    for i, curr_data_famacha in enumerate(data_famacha_list):
        try:
            data_famacha_copy = data_famacha_list
            result = get_training_data(csv_df, csv_median, csv_mean, curr_data_famacha, i, data_famacha_copy,
                                       data_famacha_dict, weather_data,
                                       days_before_famacha_test)
        except KeyError as e:
            result = None
            print(e)

        if result is None:
            continue

        # activity_resampled, herd_resampled = resample_traces(result["activity"], result["median"])
        is_valid, reason = is_activity_data_valid(result["activity_raw"])
        # print("sample is valid=", is_valid)
        # result["activity"] = activity_resampled.tolist()
        result['is_valid'] = is_valid
        if result["famacha_score"] < 0 or result["previous_famacha_score1"] < 0:
            result['is_valid'] = False
            reason = 'missing_f'

        result['reason'] = reason
        results.append(result)

    print("processing file %d done" % idx)
    process_famacha_var(results)
    exporting_data_info_to_txt(result_output_dir, results, thresh_i, thresh_z2n, animal_id)

    print("*******************************************")
    print("computing set for file %d..." % idx)
    for idx in range(len(results)):
        result = results[idx]

        if not result['is_valid']:
            continue

        # activity_array = np.array(result["activity"])
        # activity_anscombe = np.array(anscombe_list(activity_array))
        # activity_norml2 = normalize(activity_array[:, np.newaxis], axis=0).ravel()
        # 
        # result["activity"] = activity_norml2.tolist()

        if create_activity_graph_enabled:
            sub_sub_folder = str(result["is_valid"]) + "/"
            filename_graph = create_filename(result)

            tag = "11"
            if result["famacha_score_increase"]:
                tag = "12"
            create_activity_graph(result_output_dir, tag + "_" + result['reason'] + "_" + str(result["animal_id"]),
                                  result["activity"], result_output_dir, filename_graph,
                                  title=create_graph_title(result, "time"),
                                  sub_sub_folder=sub_sub_folder)

        # print("result valid %d/%dfor %s." % (idx, len(results), str(result["animal_id"])))
        # coef_shape, cwt, coef, freqs, indexes_cwt, scales, delta_t, wavelet_type, coi = compute_cwt(result["activity"])
        #
        # result["cwt"] = cwt
        # result["coef_shape"] = coef_shape
        # result["indexes_cwt"] = indexes_cwt
        #
        # create_hd_cwt_graph(result_output_dir, coef, len(cwt), result_output_dir, tag+"_"+filename_graph, title=create_graph_title(result, "freq"),
        #                     sub_sub_folder=sub_sub_folder, freqs=freqs)

        create_training_sets(result_output_dir, result, result_output_dir, days_before_famacha_test, farm_id, thresh_i, thresh_z2n)
        results[idx] = None
        # gc.collect()
        # print("create_activity_graph done.")

    print("computing set for file %d done." % idx)
    return


def exporting_data_info_to_txt(output_dir, results, thresh_i, thresh_z2n, animal_id):
    print("exporting_data_info_to_txt.")
    total_sample_11 = 0
    total_sample_12 = 0

    nan_sample_11 = 0
    nan_sample_12 = 0

    usable_11 = 0
    usable_12 = 0

    for item in results:
        if item['famacha_score_increase']:
            total_sample_12 += 1
        if not item['famacha_score_increase']:
            total_sample_11 += 1
        if item['famacha_score_increase'] and not item['is_valid']:
            nan_sample_12 += 1
        if not item['famacha_score_increase'] and not item['is_valid']:
            nan_sample_11 += 1

        if item['famacha_score_increase'] and item['is_valid']:
            usable_12 += 1
        if not item['famacha_score_increase'] and item['is_valid']:
            usable_11 += 1

    filename = "%s/%s_result_interpol_%d_zeros_%d.txt" % (output_dir, animal_id, thresh_i, thresh_z2n)
    # purge_file(filename)
    report = "Total samples = %d\n1 -> 1 = %d\n1 -> 2 = %d\nNan samples: \n1 -> 1 = %d\n1 -> 2 = %d\nUsable: \n1 " \
             "-> 1 = %d\n1 -> 2 = %d\n" % (total_sample_11+total_sample_12, total_sample_11, total_sample_12, nan_sample_11,
                             nan_sample_12, usable_11, usable_12)

    with open(filename, 'a') as outfile:
        outfile.write(report)
        outfile.write('\n')
        outfile.close()
    return total_sample_11, total_sample_12, nan_sample_11, nan_sample_12, usable_11, usable_12


def exporting_data_info_to_txt_final(output_dir, farm_id, n_days_before_famacha, thresh_i, thresh_z2n, total, total_sample_11, total_sample_12, nan_sample_11, nan_sample_12, usable_11, usable_12):
    print("FINAL")
    print("exporting_data_info_to_txt_final.")
    try:
        pathlib.Path(output_dir).mkdir(parents=True)
    except Exception as e:
        print(e)
    print("output_dir=", output_dir)
    print("total_sample_11=", total_sample_11)
    print("total_sample_12=", total_sample_12)

    filename_final = "%s/11_%d_12_%d_%s_result_days_%d_interpol_%d_zeros_%d.txt" % (output_dir, usable_11, usable_12, farm_id, int(n_days_before_famacha), int(thresh_i),int(thresh_z2n))
    print(filename_final)
    #purge_file(filename_final)
    report = "Total samples = %d\n1 -> 1 = %d\n1 -> 2 = %d\nNan samples: \n1 -> 1 = %d\n1 -> 2 = %d\nUsable: \n1 " \
             "-> 1 = %d\n1 -> 2 = %d\n" % (total, total_sample_11, total_sample_12, nan_sample_11,
                             nan_sample_12, usable_11, usable_12)

    print("filename_final=", filename_final)
    with open(filename_final, 'a') as outfile:
        outfile.write(report)
        outfile.write('\n')
        outfile.close()
    print(report)

def parse_csv_db_name(path):
    split = path.split('/')[-1].split('_')
    farm_id = split[6] + "_" + split[7]
    threshold_interpol = int(split[2])
    threshold_zeros2nan = int(split[4])
    return farm_id, threshold_interpol, threshold_zeros2nan


if __name__ == '__main__':
    print("args: output_dir csv_db_dir_path famacha_file_path n_days_before_famacha resampling_resolution enable_graph_output n_process")
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
        csv_db_dir_path = sys.argv[2]
        famacha_file_path = sys.argv[3]
        n_days_before_famacha = int(sys.argv[4])
        # resampling_resolution = sys.argv[5]
        enable_graph_output = sys.argv[5].lower() == 'true'
        n_process = int(sys.argv[6])
        print("N_PROCESS=", n_process)
        print("enable_graph=", enable_graph_output)
    else:
        exit(-1)

    print("output_dir=", output_dir)
    print("csv_db_path=", csv_db_dir_path)
    print("famacha_file_path=", famacha_file_path)
    print("n_days_before_famacha=", n_days_before_famacha)

    files = glob.glob(csv_db_dir_path+"/*csv")
    files = [file.replace("\\", '/') for file in files]
    print("found %d files."% len(files))

    if len(files) == 0:
        print("no files in %s" % csv_db_dir_path)
        exit(-1)

    file_median = None
    for idx, file in enumerate(files):
        if 'median' in file:
            file_median = file
            break
    file_mean = None
    for idx, file in enumerate(files):
        if 'mean' in file:
            file_mean = file
            break

    famacha_data = get_famacha_data(famacha_file_path)

    MULTI_THREADING_ENABLED = (n_process > 0)
    print("MULTI_THREADING_ENABLED=", MULTI_THREADING_ENABLED)

    farm_id, thresh_i, thresh_z2n = parse_csv_db_name(files[0])

    try:
        csv_median = load_db_from_csv(file_median)
    except ValueError as e:
        print("missing median file!")
        csv_median = None
        sys.exit(-1)
    try:
        csv_mean = load_db_from_csv(file_mean)
    except ValueError as e:
        print("missing mean file!")
        csv_mean = None
        sys.exit(-1)

    if not os.path.exists(output_dir):
        print("mkdir", output_dir)
        os.makedirs(output_dir)

    result_output_dir = "%s/%s_famachadays_%d_threshold_interpol_%d_threshold_zero2nan_%d" % (
        output_dir, farm_id, n_days_before_famacha, thresh_i, thresh_z2n)
    total_sample_11 = []
    total_sample_12 = []
    nan_sample_11 = []
    nan_sample_12 = []
    usable_11 = []
    usable_12 = []
    total = []
    if MULTI_THREADING_ENABLED:
        pool = Pool(processes=n_process)
        for idx, file in enumerate(files):
            if 'median' in file or 'mean' in file:
                continue
            pool.apply_async(process_day, (enable_graph_output, result_output_dir, csv_median, csv_mean,
                                           idx, thresh_i, thresh_z2n, n_days_before_famacha, farm_id, file, famacha_data,))
        pool.close()
        pool.join()
        pool.terminate()
        print("multithreaded loop finished! reading results...")
    else:
        for idx, file in enumerate(files):
            if 'median' in file or 'mean' in file:
                continue
            process_day(enable_graph_output, result_output_dir, csv_median, csv_mean, idx, thresh_i,
                        thresh_z2n, n_days_before_famacha, farm_id, file, famacha_data)

    files_txt = glob.glob(result_output_dir + "/*.txt")
    for file in files_txt:
        with open(file) as f:
            lines = [line.rstrip() for line in f]
            total.append(int(lines[0].split('=')[1].strip()))
            total_sample_11.append(int(lines[1].split('=')[1].strip()))
            total_sample_12.append(int(lines[2].split('=')[1].strip()))
            nan_sample_11.append(int(lines[4].split('=')[1].strip()))
            nan_sample_12.append(int(lines[5].split('=')[1].strip()))
            usable_11.append(int(lines[7].split('=')[1].strip()))
            usable_12.append(int(lines[8].split('=')[1].strip()))

    exporting_data_info_to_txt_final(result_output_dir, farm_id, n_days_before_famacha, thresh_i, thresh_z2n, sum(total), sum(total_sample_11), sum(total_sample_12),
                                     sum(nan_sample_11), sum(nan_sample_12), sum(usable_11), sum(usable_12))


