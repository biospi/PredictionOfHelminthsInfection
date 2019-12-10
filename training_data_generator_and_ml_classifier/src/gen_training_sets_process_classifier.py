import base64
import json
import json
import math
import os
import os.path
import pathlib
import shutil
import statistics
import sys
import time
from datetime import datetime, timedelta
import xlsxwriter
import dateutil.relativedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymysql
import pywt
import xlrd
from ipython_genutils.py3compat import xrange
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, LabelBinarizer
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, accuracy_score, make_scorer, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, StratifiedKFold
import scikitplot as skplt
from matplotlib.lines import Line2D
from mlxtend.plotting import plot_decision_regions
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import matplotlib.font_manager
import matplotlib.patches as mpatches
import pycwt as wavelet
from matplotlib.colors import LogNorm
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from os import listdir
from os.path import isfile, join
import tzlocal

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

import os
os.environ['R_HOME'] = 'C:\Program Files\R\R-3.6.1' #path to your R installation
os.environ['R_USER'] = 'C:\\Users\\fo18103\\AppData\\Local\Continuum\\anaconda3\Lib\site-packages\\rpy2' #path depends on where you installed Python. Mine is the Anaconda distribution

import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr, isinstalled
utils = importr('utils')
R = robjects.r

#need to be installed from Rstudio or other package installer
print('e1071', isinstalled('e1071'))
print('rgl', isinstalled('rgl'))
print('misc3d', isinstalled('misc3d'))
print('plot3D', isinstalled('plot3D'))
print('plot3Drgl', isinstalled('plot3Drgl'))

if not isinstalled('e1071'):
    utils.install_packages('e1071')
if not isinstalled('rgl'):
    utils.install_packages('rgl')
if not isinstalled('misc3d'):
    utils.install_packages('misc3d')
if not isinstalled('plot3D'):
    utils.install_packages('plot3D')
if not isinstalled('plot3Drgl'):
    utils.install_packages('plot3Drgl')

e1071 = importr('e1071')
rgl = importr('rgl')
misc3d = importr('misc3d')
plot3D = importr('plot3D')
plot3Drgl = importr('plot3Drgl')

print(rpy2.__version__)

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 5)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

ts = time.time()
run_timestamp = datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
print(run_timestamp)
RESULT_FILE_HEADER = "accuracy_cv,accuracy_list,precision_true,precision_false," \
                     "recall_true,recall_false,fscore_true,fscore_false,support_true,support_false," \
                     "class_true_count,class_false_count,skipped_class_true,skipped_class_false," \
                     "fold,resolution," \
                     "days_before_test,threshold_nan,threshold_zeros,processing_time,sample_count,set_size," \
                     "file_path,input,classifier, decision_bounderies_file"

RESULT_FILE_HEADER_SIMPLIFIED = "classifier, accuracy,specificity,recall,precision,fscore,days,resolution,inputs"

skipped_class_false, skipped_class_true = -1, -1


def purge_file(filename):
    print("purge %s..." % filename)
    try:
        os.remove(filename)
    except FileNotFoundError:
        print("file not found.")


def generate_table_from_xlsx(path):
    book = xlrd.open_workbook(path)
    sheet = book.sheet_by_index(1)
    data = {}
    for row_index in xrange(0, sheet.nrows):
        row_values = [sheet.cell(row_index, col_index).value for col_index in xrange(0, sheet.ncols)]
        if row_index == 2:
            time_w = list(filter(None, row_values))
            # print(time_w)

        if row_index == 3:
            idx_w = [index for index, value in enumerate(row_values) if value == "WEIGHT"]
            idx_c = [index for index, value in enumerate(row_values) if value == "FAMACHA"]

        chunks = []
        if row_index > 4:
            for i in xrange(0, len(idx_w)):
                if row_values[1] is '':
                    continue
                s = "40101310%s" % row_values[1]
                serial = int(s.split('.')[0])
                chunks.append([time_w[i], row_values[idx_c[i]], serial, row_values[idx_w[i]]])
            if len(chunks) != 0:
                data[serial] = chunks
    # print(data)
    return data


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
        print("could not find weather data!", e)

    return temp, humidity


def get_previous_famacha_score(serial_number, famacha_test_date, data_famacha, curr_score):
    previous_score = None
    list = data_famacha[serial_number]
    for i in range(1, len(list)):
        item = list[i]
        if item[0] == famacha_test_date:
            try:
                previous_score = int(list[i - 1][1])
            except ValueError as e:
                previous_score = curr_score
                # print(e)
            break
    return previous_score


def pad(a, N):
    a += [-1] * (N - len(a))
    return a


def connect_to_sql_database(db_server_name="localhost", db_user="axel", db_password="Mojjo@2015",
                            db_name="south_africa",
                            char_set="utf8mb4", cusror_type=pymysql.cursors.DictCursor):
    # print("connecting to db %s..." % db_name)
    global sql_db
    sql_db = pymysql.connect(host=db_server_name, user=db_user, password=db_password,
                             db=db_name, charset=char_set, cursorclass=cusror_type)
    return sql_db


def execute_sql_query(query, records=None, log_enabled=False):
    try:
        sql_db = connect_to_sql_database()
        cursor = sql_db.cursor()
        if records is not None:
            print("SQL Query: %s" % query, records)
            cursor.executemany(query, records)
        else:
            if log_enabled:
                print("SQL Query: %s" % query)
            cursor.execute(query)
        rows = cursor.fetchall()
        for row in rows:
            if log_enabled:
                print("SQL Answer: %s" % row)
        return rows
    except Exception as e:
        print("Exeception occured:{}".format(e))


def get_elapsed_time_string(time_initial, time_next):
    dt1 = datetime.fromtimestamp(time_initial)
    dt2 = datetime.fromtimestamp(time_next)
    rd = dateutil.relativedelta.relativedelta(dt2, dt1)
    return '%02d:%02d:%02d:%02d' % (rd.days, rd.hours, rd.minutes, rd.seconds)


def anscombe(value):
    return 2 * math.sqrt(value + (3 / 8))

# def normalize_activity_array_ascomb(activity):
#     result = []
#     for i in range(0, len(activity)):
#         try:
#             result.append({'timestamp': activity[i]['timestamp'], 'serial_number': activity[i]['serial_number'],
#                            'first_sensor_value': ascombe(activity[i]['first_sensor_value'])})
#         except (ValueError, TypeError) as e:
#             print('error while normalize_activity_array_ascomb', e)
#             result.append({'timestamp': activity[i]['timestamp'], 'serial_number': activity[i]['serial_number'],
#                            'first_sensor_value': None})
#     return result


def anscombe_list(activity):
    return [anscombe(x) if x is not None else None for x in activity]


def anscombe_log_list(activity):
    return [math.log(anscombe(x)) if x is not None else None for x in activity]


def normalize_histogram_mean_diff(activity_mean, activity, log=False):
    scale = [0 for _ in range(0, len(activity))]
    idx = []
    for n, a in enumerate(activity):
        if a is None or a <= 0:
            continue
        if activity_mean[n] is None:
            continue
        if log:
            r = (int(activity_mean[n]) - int(a))
        else:
            r = (int(activity_mean[n]) / int(a))

        scale[n] = r
        idx.append(n)
    median = math.fabs(statistics.median(sorted(set(scale))))
    #print(scale)
    for i in idx:
        activity[i] = activity[i] * median
    return activity


def get_period(curr_data_famacha, days_before_famacha_test):
    famacha_test_date = time.strptime(curr_data_famacha[0], "%d/%m/%Y")
    famacha_test_date_epoch_s = str(time.mktime(famacha_test_date)).split('.')[0]
    famacha_test_date_epoch_before_s = str(time.mktime((datetime.fromtimestamp(time.mktime(famacha_test_date)) -
                                                        timedelta(days=days_before_famacha_test)).timetuple())).split(
        '.')[0]
    return famacha_test_date_epoch_s, famacha_test_date_epoch_before_s


def get_training_data(curr_data_famacha, data_famacha_dict, weather_data, resolution, days_before_famacha_test,
                      expected_sample_count, farm_sql_table_id=None):
    # print("generating new training pair....")
    famacha_test_date = datetime.fromtimestamp(time.mktime(time.strptime(curr_data_famacha[0], "%d/%m/%Y"))).strftime(
        "%d/%m/%Y")
    try:
        famacha_score = int(curr_data_famacha[1])
        animal_weight = int(curr_data_famacha[3])
    except ValueError as e:
        print("error while parsing famacha score!", e)
        return

    animal_id = curr_data_famacha[2]
    # find the activity data of that animal the n days before the test
    date1, date2 = get_period(curr_data_famacha, days_before_famacha_test)
    print("getting activity data for test on the %s for %d. collecting data %d days before resolution is %s..." % (
        famacha_test_date, animal_id, days_before_famacha_test, resolution))
    rows_activity = execute_sql_query("SELECT timestamp, serial_number, first_sensor_value FROM %s_resolution_%s"
                                      " WHERE timestamp BETWEEN %s AND %s AND serial_number = %s" %
                                      (farm_sql_table_id, resolution, date2, date1,
                                       str(animal_id)))

    rows_herd = execute_sql_query(
        "SELECT timestamp, serial_number, first_sensor_value FROM %s_resolution_%s WHERE serial_number=%s AND timestamp BETWEEN %s AND %s" %
        (farm_sql_table_id, resolution, 50000000000, date2, date1))

    # activity_mean = normalize_activity_array_ascomb(rows_mean)
    herd_activity_list = [(x['first_sensor_value']) for x in rows_herd]
    activity_list = [(x['first_sensor_value']) for x in rows_activity]

    if len(rows_activity) != expected_sample_count:
        # filter out missing activity data
        print("absent activity records. skip.", "found %d" % len(rows_activity), "expecred %d" % expected_sample_count)
        return

    # data_activity = normalize_activity_array_ascomb(data_activity)
    herd_activity_list = anscombe_log_list(herd_activity_list)
    activity_list = anscombe_log_list(activity_list)

    # herd_activity_list = anscombe_list(herd_activity_list)
    # activity_list = anscombe_list(activity_list)

    activity_list = normalize_histogram_mean_diff(herd_activity_list, activity_list, log=True)

    # print("mapping activity to famacha score progress=%d/%d ..." % (i, len(data_famacha_flattened)))
    idx = 0
    indexes = []
    timestamp_list = []
    humidity_list = []
    # activity_list = []
    temperature_list = []
    dates_list_formated = []

    for j, data_a in enumerate(rows_activity):
        # transform date in time for comparaison
        curr_datetime = datetime.utcfromtimestamp(int(data_a['timestamp']))
        timestamp = time.strptime(curr_datetime.strftime('%d/%m/%Y'), "%d/%m/%Y")
        # if not weather_data:
        #     temp, humidity = 0, 0
        # else:
        #     temp, humidity = get_temp_humidity(curr_datetime, weather_data)

        temp, humidity = 0, 0
        # activity_list.append(data_activity_o[j])
        indexes.append(idx)
        timestamp_list.append(timestamp)
        temperature_list.append(temp)
        humidity_list.append(humidity)
        dates_list_formated.append(datetime.utcfromtimestamp(int(data_a['timestamp'])).strftime('%d/%m/%Y %H:%M'))
        idx += 1

    # activity_list = [a_i - b_i if a_i is not None and b_i is not None else None for a_i, b_i in
    #                  zip(activity_list, activity_mean)]

    # activity_list_np = np.array(activity_list, dtype=np.float)
    # activity_mean_np = np.array(activity_mean, dtype=np.float)

    # activity_list_np = np.divide(activity_list_np, activity_mean_np)
    # activity_list_np = np.where(np.isnan(activity_list_np), None, activity_list_np)
    # activity_list = activity_list_np.tolist()

    previous_famacha_score = get_previous_famacha_score(animal_id, famacha_test_date, data_famacha_dict, famacha_score)
    indexes.reverse()

    # herd_activity_list = anscombe_list(herd_activity_list)
    # activity_list = anscombe_list(activity_list)

    data = {"famacha_score_increase": False, "famacha_score": famacha_score, "animal_weight": animal_weight,
            "previous_famacha_score": previous_famacha_score, "animal_id": animal_id,
            "date_range": [time.strftime('%d/%m/%Y', time.localtime(int(date1))),
                           time.strftime('%d/%m/%Y', time.localtime(int(date2)))],
            "indexes": indexes, "activity": activity_list,
            "temperature": temperature_list, "humidity": humidity_list, "herd": herd_activity_list
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
            if count_skipped:
                skipped_class_false += 1
            continue
        if curr_data["famacha_score"] == 1 and next_data["famacha_score"] == 2:
            print("famacha score changed from 1 to 2. creating new set...")
            next_data["famacha_score_increase"] = True
            if count_skipped:
                skipped_class_true += 1
        if curr_data["famacha_score"] == 2 and next_data["famacha_score"] == 1:
            if count_skipped:
                skipped_class_false += 1
            continue
    return skipped_class_false, skipped_class_true


def find_minimum_delay_beteen_test(data_famacha_flattened):
    time_between_tests = []
    for i, current_data in enumerate(data_famacha_flattened[:-1]):
        next_data = data_famacha_flattened[i + 1]
        if current_data[2] == next_data[2]:
            d2 = datetime.strptime(next_data[0], '%d/%m/%Y')
            d1 = datetime.strptime(current_data[0], '%d/%m/%Y')
            diff = (d2 - d1)
            time_between_tests.append(diff.days)
            if (current_data[1] == 3):
                print(diff.days, current_data, next_data)
    print("minimum delay between famacha tests is %d days." % min(time_between_tests))


def get_expected_sample_count(resolution, days_before_test):
    expected_sample_n = None
    if resolution == "min":
        expected_sample_n = (24 * 60) * days_before_test
    if resolution == "5min":
        expected_sample_n = ((24 * 60) / 5) * days_before_test
    if resolution == "10min":
        expected_sample_n = ((24 * 60) / 10) * days_before_test
    if resolution == "hour":
        expected_sample_n = (24 * days_before_test) - 1  # todo fix
    if resolution == "day":
        expected_sample_n = days_before_test - 1  # todo fix

    expected_sample_n = expected_sample_n + 1
    print("expected sample count is %d." % expected_sample_n)
    return expected_sample_n


def contains_negative(list):
    for v in list:
        if v is None:
            continue
        if v < 0:
            print("negative number found!", v)
            exit()
            return True
    return False


def is_activity_data_valid(activity, threshold_nan_coef, threshold_zeros_coef):
    nan_threshold = len(activity) / threshold_nan_coef
    zeros_threshold = len(activity) / threshold_zeros_coef
    nan_count = activity.count(None)
    zeros_count = activity.count(0)
    # print(nan_count, zeros_count, nan_threshold, zeros_threshold)
    if nan_count > nan_threshold or zeros_count > zeros_threshold or contains_negative(activity):
        return False, nan_threshold, zeros_threshold
    return True, nan_threshold, zeros_threshold


def multiple(m, n):
    return list(range(n, (m * n) + 1, n))


def even_list(n):
    result = [1]
    for num in range(2, n * 2 + 1, 2):
        result.append(num)
    del result[-1]
    return np.asarray(result, dtype=np.int32)


def interpolate(input_activity):
    try:
        i = np.array(input_activity, dtype=np.float)
        s = pd.Series(i)
        s = s.interpolate(method='cubic', limit_direction='both')
        s = s.interpolate(method='linear', limit_direction='both')
        return s.tolist()
    except ValueError as e:
        print(e)
        return input_activity


def create_activity_graph(activity, folder, filename, title=None, sub_folder='training_sets_time_domain_graphs'):
    activity = [0 if x is None else x for x in activity]
    fig = plt.figure()
    plt.bar(range(0, len(activity)), activity)
    fig.suptitle(title, x=0.5, y=.95, horizontalalignment='center', verticalalignment='top', fontsize=10)
    path = "%s/%s" % (folder, sub_folder)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    fig.savefig('%s/%s' % (path, filename))


def create_cwt_graph(coef, freqs, lenght, folder, filename, title=None):
    time = [x for x in range(0, lenght)]
    fig = plt.figure()
    plt.pcolormesh(time, freqs, coef)
    fig.suptitle(title, x=0.5, y=.95, horizontalalignment='center', verticalalignment='top', fontsize=10)
    path = "%s/training_sets_cwt_graphs" % folder
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    fig.savefig('%s/%s' % (path, filename))
    coef_f = coef.flatten().tolist()
    create_activity_graph(coef_f, folder, "flat_%s" % filename, title, sub_folder='training_sets_cwt_graphs')


def create_hd_cwt_graph(coefs, folder, filename, title=None):
    fig = plt.figure()
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.imshow(coefs)
    ax.set_yscale('log')
    path = "%s/training_sets_cwt_graphs" % folder
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    fig.savefig('%s/%s' % (path, filename))
    exit()
    # fig.savefig('%s/%s' % (path, filename))


def compute_hd_cwt(activity):
    print("compute_cwt...")
    # t, activity = dummy_sin()
    activity = interpolate(activity)
    num_steps = len(activity)
    x = np.arange(num_steps)
    y = np.asarray(activity, dtype=np.float)
    # y[np.isnan(y)] = -1

    delta_t = x[1] - x[0]
    scales = np.arange(1, num_steps + 1) / 1
    freqs = 1 / (wavelet.Morlet().flambda() * scales)
    wavelet_type = 'morlet'

    coefs, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(y, delta_t, wavelet=wavelet_type, freqs=freqs)

    # iwave = wavelet.icwt(coefs, scales, delta_t, wavelet=wavelet_type)
    # plt.plot(iwave)
    # plt.show()
    # plt.plot(activity)
    # plt.show()
    #
    # plt.matshow(coefs.real)
    # plt.show()
    # exit()
    cwt = [element for tupl in coefs.real for element in tupl]
    print("cwt len is %d" % len(cwt))
    # indexes = list(range(len(cwt)))
    # indexes.reverse()
    indexes = []
    return cwt, coefs.real, freqs, indexes, scales, delta_t, wavelet_type


def compute_cwt(activity):
    name = 'gaus8'
    w = pywt.ContinuousWavelet(name)
    scales = even_list(40)
    sampling_frequency = 1 / 60
    sampling_period = 1 / sampling_frequency
    activity_i = interpolate(activity)
    coef, freqs = pywt.cwt(np.asarray(activity_i), scales, w, sampling_period=sampling_period)
    cwt = [element for tupl in coef for element in tupl]
    indexes = list(range(len(cwt)))
    indexes.reverse()
    return cwt, coef, freqs, indexes


def create_filename(data):
    filename = "%d_%s_%s_%fsd_pvfs%d_zt%d_nt%d.png" % (data["animal_id"], data["date_range"][0], data["date_range"][1],
                                                       data["famacha_score"], data["previous_famacha_score"],
                                                       data["nan_threshold"],
                                                       data["zeros_threshold"])
    return filename.replace('/', '-')


def process_weight(activity, cwt):
    weight_map = np.zeros(cwt.shape, dtype=float)
    # set first raw of the map according to time series values
    for j in range(0, len(activity)):
        if activity[j] is None:
            weight_map[cwt.shape[0] - 1, j] = 0
        else:
            weight_map[cwt.shape[0] - 1, j] = 1
    # compute weights
    j = 0
    rows = []
    mult_of_2 = multiple(weight_map.shape[0], 2)
    # print(mult_of_2)
    for k in list(reversed(range(0, weight_map.shape[0]))):
        row = weight_map[k, :]
        # print(row.tolist())
        rows.append(row.tolist())
        cpt = 0
        values = []
        idxs = []
        for n, elem in enumerate(row):
            idxs.append(n)
            values.append(elem)
            cpt += 1
            if cpt == mult_of_2[j]:
                if j > 0:
                    s = sum(list(set(values)))
                else:
                    s = sum(values)
                for idxs_i in idxs:
                    weight_map[k - 1, idxs_i] = s
                cpt = 0
                values = []
                idxs = []

        if len(values) > 0:
            s = sum(list(set(values)))
            for idxs_i in idxs:
                weight_map[k - 1, idxs_i] = s
        j += 1
    weight_map_final = np.array(list(reversed(rows)))
    return weight_map_final.flatten().tolist()


def create_training_set(result, dir, options=[]):
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
        training_set.append(result["previous_famacha_score"])
        option = option + "previous_score_"
    training_set.append(result["famacha_score_increase"])
    training_set.append(len(training_set))
    training_set.extend(result["date_range"])
    training_set.append(result["animal_id"])
    path = "%s/training_sets" % dir
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    filename = "%s/%s.data" % (path, option)
    # print(len(training_set))

    # map(lambda x: str(x) if x is not None else 'NaN', training_set)

    # training_str_flatten = ','.join(training_set)
    training_str_flatten = str(training_set).strip('[]').replace(' ', '').replace('None', 'NaN')
    print("set size is %d, %s.....%s" % (
        len(training_set), training_str_flatten[0:50], training_str_flatten[-50:]))
    with open(filename, 'a') as outfile:
        outfile.write(training_str_flatten)
        outfile.write('\n')
    return filename, options


# def create_training_sets(data, dir_path):
#     training_file_path_hum_temp_activity, options_hum_temp_activity = create_training_set(data, dir_path, options=["activity", "humidity", "temperature"])
#     training_file_path_hum_temp_cwt, options_hum_temp_cwt = create_training_set(data, dir_path, options=["cwt", "humidity", "temperature"])
#     training_file_path_activity, options_activity = create_training_set(data, dir_path, options=["activity"])
#     training_file_path_cwt, options_cwt = create_training_set(data, dir_path, options=["cwt"])
#
#     return [
#         {"path": training_file_path_hum_temp_activity, "options": options_hum_temp_activity},
#         {"path": training_file_path_hum_temp_cwt, "options": options_hum_temp_cwt},
#         {"path": training_file_path_activity, "options": options_activity},
#         {"path": training_file_path_cwt, "options": options_cwt}
#     ]


def create_training_sets(data, dir_path):
    # training_file_path_td, options1 = create_training_set(data, dir_path,
    #                                                       options=["activity", "indexes", "humidity", "temperature"])
    # training_file_path_td1, options11 = create_training_set(data, dir_path,
    #                                                         options=["activity", "humidity", "temperature"])
    # training_file_path_fd, options2 = create_training_set(data, dir_path,
    #                                                       options=["cwt", "cwt_weight", "indexes_cwt", "humidity",
    #                                                                "temperature"])
    # training_file_path_fd1, options21 = create_training_set(data, dir_path,
    #                                                         options=["cwt", "cwt_weight", "humidity", "temperature"])
    # training_file_path_fd12, options22 = create_training_set(data, dir_path, options=["cwt", "humidity", "temperature"])
    # training_file_path_fd123, options223 = create_training_set(data, dir_path, options=["cwt", "humidity"])
    # training_file_path_fd1232, options2232 = create_training_set(data, dir_path, options=["cwt", "temperature"])
    # training_file_path_td144, options114 = create_training_set(data, dir_path, options=["activity", "temperature"])
    # training_file_path_td14, options1142 = create_training_set(data, dir_path, options=["activity", "humidity"])
    # training_file_path_temp, options3 = create_training_set(data, dir_path, options=["temperature"])
    # training_file_path_hum, options4 = create_training_set(data, dir_path, options=["humidity"])
    # training_file_path_hum_temp, options5 = create_training_set(data, dir_path, options=["humidity", "temperature"])
    training_file_path_hum_temp_activity, options7 = create_training_set(data, dir_path, options=["activity"])
    training_file_path_hum_temp_cwt, options8 = create_training_set(data, dir_path, options=["cwt"])

    return [
        #{"path": training_file_path_td, "options": options1},
            # {"path": training_file_path_fd, "options": options2},
            # {"path": training_file_path_temp, "options": options3},
            # {"path": training_file_path_hum, "options": options4},
            # {"path": training_file_path_hum_temp, "options": options5},
            {"path": training_file_path_hum_temp_activity, "options": options7},
            {"path": training_file_path_hum_temp_cwt, "options": options8}
            #{"path": training_file_path_td1, "options": options11},
            # {"path": training_file_path_fd1, "options": options21},
            #{"path": training_file_path_fd123, "options": options223},
            #{"path": training_file_path_fd1232, "options": options2232},
            #{"path": training_file_path_td144, "options": options114},
            #{"path": training_file_path_td14, "options": options1142},
            #{"path": training_file_path_fd12, "options": options22}
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
            data["previous_famacha_score"],
            str(data["famacha_score_increase"]))
    if domain == "freq":
        idxs_1 = ','.join([str(int(x)) for x in data["indexes_cwt"][0:1]])
        idxs_2 = ','.join([str(int(x)) for x in data["indexes_cwt"][-1:]])
        cwt_1 = ','.join([str(int(x)) for x in data["cwt"][0:1]])
        cwt_2 = ','.join([str(int(x)) for x in data["cwt"][-1:]])
        return "[cwt:[%s...%s],idxs:[%s...%s],h:[%s...%s],t:[%s...%s],fs:%d,pfs:%d,%s]" % (
            cwt_1, cwt_2, idxs_1, idxs_2, hum_1, hum_2,
            temp_1, temp_2, data["famacha_score"],
            data["previous_famacha_score"],
            str(data["famacha_score_increase"]))


def init_result_file(dir, simplified_results=False):
    path = "%s/analysis" % dir
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    filename = "%s/results_simplified.csv" % path if simplified_results else "%s/results.csv" % path
    with open(filename, 'a') as outfile:
        outfile.write(RESULT_FILE_HEADER_SIMPLIFIED) if simplified_results else outfile.write(RESULT_FILE_HEADER)
        outfile.write('\n')
    outfile.close()
    return filename


def append_simplified_result_file(filename, classifier_name, accuracy, specificity, recall, precision, fscore,
                                  days_before_test, resolution, options):
    data = "%s, %.2f,%.2f,%.2f,%.2f,%.2f,%d,%s,%s" % (
    classifier_name.replace(',', ':').replace(' 10FCV', ''), accuracy, specificity, recall, precision, fscore,
    days_before_test, resolution, options)
    with open(filename, 'a') as outfile:
        outfile.write(data)
        outfile.write('\n')
    outfile.close()


def append_result_file(filename, cross_validated_score, scores, precision_true, precision_false,
                       recall_true, recall_false, fscore_true, fscore_false, support_true, support_false,
                       class_true_count, class_false_count, fold,
                       resolution, days_before_test, threshold_nan, threshold_zeros,
                       processing_time,
                       sample_count, set_size, training_file, options, kernel,
                       db_path):
    global skipped_class_false
    global skipped_class_true
    scores_s = ' '.join([str(x) for x in scores])
    print('cross_validated_score', type(cross_validated_score), cross_validated_score)
    print('scores', type(scores), scores)
    print('scores_s', type(scores_s), scores_s)
    print('precision_true', type(precision_true), precision_true)
    print('precision_false', type(precision_false), precision_false)
    print('recall_true', type(recall_true), recall_true)
    print('recall_false', type(recall_false), recall_false)
    print('fscore_true', type(fscore_true), fscore_true)
    print('fscore_false', type(fscore_false), fscore_false)
    print('support_true', type(support_true), support_true)
    print('support_false', type(support_false), support_false)
    print('class_true_count', type(class_true_count), class_true_count)
    print('class_false_count', type(class_false_count), class_false_count)
    print('fold', type(fold), fold)
    print('resolution', type(resolution), resolution)
    print('days_before_test', type(days_before_test), days_before_test)
    print('threshold_nan', type(threshold_nan), threshold_nan)
    print('threshold_zeros', type(threshold_zeros), threshold_zeros)
    print('processing_time', type(processing_time), processing_time)
    print('sample_count', type(sample_count), sample_count)
    print('set_size', type(set_size), set_size)
    print('training_file', type(training_file), training_file)
    print('options', type(options), options)
    print('kernel', type(kernel), kernel)
    print('db_path', type(db_path), db_path)
    data = "%.15f,%s,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%d,%d,%d,%d,%d,%d,%d,%s,%d,%d,%d,%s,%d,%d,%s,%s,%s,%s" % (
        cross_validated_score, scores_s, precision_true, precision_false, recall_true, recall_false, fscore_true,
        fscore_false,
        support_true, support_false, class_true_count, class_false_count, skipped_class_true, skipped_class_false, fold,
        resolution, days_before_test, threshold_nan, threshold_zeros,
        processing_time, sample_count, set_size, training_file, '-'.join(options),
        kernel.replace(',', ':'), db_path)
    with open(filename, 'a') as outfile:
        outfile.write(data)
        outfile.write('\n')
    outfile.close()
    print(skipped_class_false, skipped_class_true)


def drange2(start, stop, step):
    numelements = int((stop - start) / float(step))
    for i in range(numelements + 1):
        yield start + i * step


def percentage(percent, whole):
    result = int((percent * whole) / 100.0)
    print("%d percent of %d is %d." % (percent, whole, result))
    return result


def parse_report(report):
    precision_true, precision_false, score = -1, -1, -1
    try:
        precision_false = report['1']['precision']
        precision_true = report['2']['precision']
        score = report['micro avg']['precision']
    except KeyError as e:
        print(report)
        print("error while parsing report", e)
    return precision_true, precision_false, score


def process_data_frame(data_frame):
    data_frame = data_frame.replace([np.inf, -np.inf], np.nan)
    data_frame = data_frame.fillna(-1)
    X = data_frame[data_frame.columns[0:data_frame.shape[1] - 1]].values
    X = normalize(X)
    X = preprocessing.MinMaxScaler().fit_transform(X)
    # print(X.shape, X)
    # print(DataFrame.from_records(X))
    y = data_frame["class"].values.flatten()
    hold_out = X.shape[0] - 0
    return X[0:hold_out, :], y[0:hold_out], X[hold_out:, :], y[hold_out:]


def process_and_split_data_frame(data_frame):
    data_frame = data_frame.fillna(-1)
    X = data_frame[data_frame.columns[0:data_frame.shape[1] - 1]].values
    X = normalize(X)
    X = preprocessing.MinMaxScaler().fit_transform(X)
    print(X.shape, X)
    # print(DataFrame.from_records(X))
    y = data_frame["class"].values.flatten()
    train_x, test_x, train_y, test_y = train_test_split(X, y, train_size=0.7, shuffle=True)
    return X, y, train_x, test_x, train_y, test_y


def get_prec_recall_fscore_support(test_y, pred_y):
    precision_recall_fscore_support_result = precision_recall_fscore_support(test_y, pred_y, average=None,
                                                                             labels=[0, 1])
    precision_false = precision_recall_fscore_support_result[0][1]
    precision_true = precision_recall_fscore_support_result[0][0]
    recall_false = precision_recall_fscore_support_result[1][1]
    recall_true = precision_recall_fscore_support_result[1][0]
    fscore_false = precision_recall_fscore_support_result[2][1]
    fscore_true = precision_recall_fscore_support_result[2][0]
    support_false = precision_recall_fscore_support_result[3][1]
    support_true = precision_recall_fscore_support_result[3][0]
    return precision_false, precision_true, recall_false, recall_true, fscore_false, fscore_true, support_false, support_true


def format_options_(options):
    return ''.join(options).replace('humidity', 'h').replace('temperature', 't').replace('activity', 'a').replace(
        'indexes', 'i')


def format_file_name(i, title, options):
    filename_i = "%d_%s_%s" % (i, title.split('\n')[0], '_'.join(options))
    filename_i = filename_i.replace(',', '-').replace(' ', '').replace(' ', '_').replace('-', '_')
    filename_i = filename_i[0:46]  # windows limits filename size by default
    filename_i = "%s.png" % filename_i
    return format_options_(filename_i)


def format_sub_folder_name(title, options):
    sub_folder = "%s_%s" % (title.split('\n')[0], '_'.join(options))
    sub_folder = sub_folder.replace(',', '-').replace(' ', '').replace(' ', '_').replace('-', '_')
    sub_folder = '_'.join(sub_folder.split('_')[0:2]) + '_' + '_'.join(options)
    return format_options_(sub_folder)


def save_roc_curve(y_test, y_probas, title, options, folder, i=0):
    # fig = plt.figure(figsize=(7, 6), dpi=100)
    # plt.title('ROC Curves %s' % title)
    split = title.split('\n')
    title = 'ROC Curves %s' % (split[0] + '\n' + split[1] + '\n' + split[-2])
    skplt.metrics.plot_roc(y_test, y_probas, title=title, title_fontsize='medium')
    path = "%s/roc_curve/%s" % (folder, format_sub_folder_name(title, options))
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    final_path = '%s/%s' % (path, format_file_name(i, title, options))
    final_path = final_path.replace('/', '\'').replace('\'', '\\').replace('\\', '/')
    print(final_path)
    plt.savefig(final_path)
    plt.close()


def plot_2D_decision_boundaries(X, y, X_test, title, clf, folder=None, options=None, i=0):
    fig = plt.figure(figsize=(8, 7), dpi=100)
    plt.subplots_adjust(top=0.80)
    scatter_kwargs = {'s': 120, 'edgecolor': None, 'alpha': 0.7}
    contourf_kwargs = {'alpha': 0.2}
    scatter_highlight_kwargs = {'s': 120, 'label': 'Test data', 'alpha': 0.7}
    plot_decision_regions(X, y, clf=clf, legend=2,
                          X_highlight=X_test,
                          scatter_kwargs=scatter_kwargs,
                          contourf_kwargs=contourf_kwargs,
                          scatter_highlight_kwargs=scatter_highlight_kwargs)
    plt.title(title)
    path = "%s/decision_boundaries_graphs/%s" % (folder, format_sub_folder_name(title, options))
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    final_path = '%s/%s' % (path, format_file_name(i, title, options))
    final_path = final_path.replace('/', '\'').replace('\'', '\\').replace('\\', '/')
    print(final_path)
    try:
        plt.savefig(final_path)
    except FileNotFoundError as e:
        print(e)
        exit()

    plt.close()
    # fig.show()
    return final_path


def plot_3D_decision_boundaries(train_x, train_y, test_x, test_y, title, clf, folder=None, options=None, i=0):
    R('r3dDefaults$windowRect <- c(0,50, 1000, 1000) ')
    R('open3d()')
    plot3ddb = R('''
       plot3ddb<-function(nnew, group, dat, kernel_, gamma_, coef_, cost_, tolerance_, probability_, test_x_, fitted_, title_, filepath){
               set.seed(12345)
               fit = svm(group ~ ., data=dat, kernel=kernel_, gamma=gamma_, coef0=coef_, cost=cost_, tolerance=tolerance_, fitted= fitted_, probability= probability_)
               x = dat[,-1]$X1
               y = dat[,-1]$X2
               z = dat[,-1]$X3
               x_test = test_x_[,-1]$X1
               y_test = test_x_[,-1]$X2
               z_test = test_x_[,-1]$X3
               i <- 1
               g = dat$group
               x_1 <- list()
               y_1 <- list()
               z_1 <- list()
               x_2 <- list()
               y_2 <- list()
               z_2 <- list()
               for(var in g){
                   if(!(x[i] %in% x_test) & !(y[i] %in% y_test)){
                       if (var == 1){
                           x_1 <- append(x_1, x[i])
                           y_1 <- append(y_1, y[i])
                           z_1 <- append(z_1, z[i])
                       }else{
                           x_2 <- append(x_2, x[i])
                           y_2 <- append(y_2, y[i])
                           z_2 <- append(z_2, z[i])
                         }
                   }
                 i <- i + 1
               }
               x_1 = as.numeric(x_1)
               y_1 = as.numeric(y_1)
               z_1 = as.numeric(z_1)
               x_2 = as.numeric(x_2)
               y_2 = as.numeric(y_2)
               z_2 = as.numeric(z_2)
               j <- 1
               g_test = test_x_$class
               x_1_test <- list()
               y_1_test <- list()
               z_1_test <- list()
               x_2_test <- list()
               y_2_test <- list()
               z_2_test <- list()
               for(var_test in g_test){
                 if (var_test == 1){
                   x_1_test <- append(x_1_test, x_test[j])
                   y_1_test <- append(y_1_test, y_test[j])
                   z_1_test <- append(z_1_test, z_test[j])
                 }else{
                   x_2_test <- append(x_2_test, x_test[j])
                   y_2_test <- append(y_2_test, y_test[j])
                   z_2_test <- append(z_2_test, z_test[j])
                 }

                 j <- j + 1
               }

               x_1_test = as.numeric(x_1_test)
               y_1_test = as.numeric(y_1_test)
               z_1_test = as.numeric(z_1_test)

               x_2_test = as.numeric(x_2_test)
               y_2_test = as.numeric(y_2_test)
               z_2_test = as.numeric(z_2_test)

               pch3d(x_2, y_2, z_2, pch = 24, bg = "#f19c51", color = "#f19c51", radius=0.4, alpha = 0.8)
               pch3d(x_1, y_1, z_1, pch = 22, bg = "#6297bb", color = '#6297bb', radius=0.4, alpha = 1)

               pch3d(x_1_test, y_1_test, z_1_test, pch = 22, bg = "#6297bb", color = 'red', radius=0.4, alpha = 0.8)
               pch3d(x_2_test, y_2_test, z_2_test, pch = 24, bg = "#f19c51", color = "red", radius=0.4, alpha = 1)

               newdat.list = lapply(test_x_[,-1], function(x) seq(min(x), max(x), len=nnew))
               newdat      = expand.grid(newdat.list)
               newdat.pred = predict(fit, newdata=newdat, decision.values=T)
               newdat.dv   = attr(newdat.pred, 'decision.values')
               newdat.dv   = array(newdat.dv, dim=rep(nnew, 3))
               grid3d(c("x", "y+", "z"))
               view3d(userMatrix = structure(c(0.850334823131561, -0.102673642337322, 
                                       0.516127586364746, 0, 0.526208400726318, 0.17674557864666, 
                                       -0.831783592700958, 0, -0.00582099659368396, 0.978886127471924, 
                                       0.20432074368, 0, 0, 0, 0, 1)))

               decorate3d(box=F, axes = T, xlab = '', ylab='', zlab='', aspect = FALSE, expand = 1.03)
               light3d(diffuse = "gray", specular = "gray")

               max_ = max(as.numeric(newdat.dv))
               min_ = min(as.numeric(newdat.dv))
               mean_ = (max_+min_)/2
               contour3d(newdat.dv, level=mean_, x=newdat.list$X1, y=newdat.list$X2, z=newdat.list$X3, add=T, alpha=0.8, plot=T, smooth = 200, color='#28b99d', color2='#28b99d')

               bgplot3d({
                         plot.new()
                         title(main = title_, line = -8, outer=F)
                         #mtext(side = 1, 'This is a subtitle', line = 4)
                         legend("bottomleft", inset=.1,
                                  pt.cex = 2,
                                  cex = 1, 
                                  bty = "n", 
                                  legend = c("Decision boundary", "Class 0", "Class 1", "Test data"), 
                                  col = c("#28b99d", "#6297bb", "#f19c51", "red"), 
                                  pch = c(15, 15,17, 1))
               })
               rgl.snapshot(filepath, fmt="png", top=TRUE)
               rgl.close()
       }''')

    nnew = test_x.shape[0]
    gamma = clf.best_params_['gamma']
    coef0 = clf.estimator.coef0
    cost = clf.best_params_['C']
    tolerance = clf.estimator.tol
    probability_ = clf.estimator.probability

    df = pd.DataFrame(train_x)
    df.insert(loc=0, column='group', value=train_y + 1)
    df.columns = ['group', 'X1', 'X2', 'X3']
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    r_dataframe = pandas2ri.py2ri(df)

    df_test = pd.DataFrame(test_x)
    df_test.insert(loc=0, column='class', value=test_y + 1)
    df_test.columns = ['class', 'X1', 'X2', 'X3']
    r_dataframe_test = pandas2ri.py2ri(df_test)

    path = "%s/decision_boundaries_graphs/%s" % (folder, format_sub_folder_name(title, options))
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    final_path = '%s/%s' % (path, format_file_name(i, title, options))
    final_path = final_path.replace('/', '\'').replace('\'', '\\').replace('\\', '/')
    print(final_path)

    plot3ddb(nnew, robjects.IntVector(train_y + 1), r_dataframe, 'radial', gamma, coef0, cost, tolerance, probability_,
             r_dataframe_test, True, title, final_path)

    # plt.savefig(final_path)
    # plt.close()
    # fig.show()
    return final_path


def reduce_lda(output_dim, X_train, X_test, y_train, y_test):
    # lda implementation require 3 input class for 2d output and 4 input class for 3d output
    print('reduce_lda', output_dim)
    if output_dim not in [2, 3]:
        raise ValueError("available dimension for features reduction are 2 and 3.")
    if output_dim == 3:
        X_train = np.vstack((X_train, np.array([np.zeros(X_train.shape[1]), np.ones(X_train.shape[1])])))
        y_train = np.append(y_train, (3, 4))
        X_test = np.vstack((X_test, np.array([np.zeros(X_test.shape[1]), np.ones(X_train.shape[1])])))
        y_test = np.append(y_test, (3, 4))
    if output_dim == 2:
        X_train = np.vstack((X_train, np.array([np.zeros(X_train.shape[1])])))
        y_train = np.append(y_train, 3)
        X_test = np.vstack((X_test, np.array([np.zeros(X_test.shape[1])])))
        y_test = np.append(y_test, 3)
    X_train = LDA(n_components=output_dim).fit_transform(X_train, y_train)
    X_test = LDA(n_components=output_dim).fit_transform(X_test, y_test)
    X_train = X_train[0:-(output_dim - 1)]
    y_train = y_train[0:-(output_dim - 1)]
    X_test = X_test[0:-(output_dim - 1)]
    y_test = y_test[0:-(output_dim - 1)]

    return X_train, X_test, y_train, y_test


def reduce_pca(output_dim, X_train, X_test, y_train, y_test):
    if output_dim not in [2, 3]:
        raise ValueError("available dimension for features reduction are 2 and 3.")
    X_train = PCA(n_components=output_dim).fit_transform(X_train)
    X_test = PCA(n_components=output_dim).fit_transform(X_test)
    return X_train, X_test, y_train, y_test


def process_fold(n, X, y, train_index, test_index, dim_reduc=None):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

    if dim_reduc is None:
        return X, y, X_train, X_test, y_train, y_test

    if dim_reduc == 'LDA':
        X_train, X_test, y_train, y_test = reduce_lda(n, X_train, X_test, y_train, y_test)

    if dim_reduc == 'PCA':
        X_train, X_test, y_train, y_test = reduce_pca(n, X_train, X_test, y_train, y_test)

    try:
        X_reduced = np.concatenate((X_train, X_test), axis=0)
        y_reduced = np.concatenate((y_train, y_test), axis=0)

        X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = X_reduced[train_index], X_reduced[
            test_index], \
                                                                           y_reduced[train_index], \
                                                                           y_reduced[test_index]
        return X_reduced, y_reduced, X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced

    except ValueError as e:
        print(e)
        return None, None, None, None, None, None


def compute_model(X, y, X_val, y_val, train_index, test_index, i, clf=None, dim=None, dim_reduc_name='', clf_name='',
                  folder=None,
                  options=None, resolution=None, enalble_2Dplot=True, enalble_3Dplot=False, enalble_ROCplot=False):
    print('dim', dim, dim_reduc_name)
    if clf_name not in ['SVC', 'MLP']:
        print(clf_name)
        raise ValueError("available classifiers are SVC and MLP.")

    X_lda, y_lda, X_train, X_test, y_train, y_test = process_fold(dim, X, y, train_index, test_index,
                                                                  dim_reduc=dim_reduc_name)

    if X_lda is None:
        return -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1

    # if dim_reduc_name != '':
    #     X_val, _, y_val, _ = reduce_lda(dim, X_val, X_val, y_val, y_val)
    # else:
    #     print('no dimentional reduction detected.')
    #
    # print(X_val)
    # print(y_val)

    clf.fit(X_train, y_train)
    print("Best estimator found by grid search:")
    # print(clf.best_estimator_)

    y_pred = clf.predict(X_test)
    # y_pred_val = clf.predict(X_val)
    y_probas = clf.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    # acc_val = accuracy_score(y_val, y_pred_val)
    print(classification_report(y_test, y_pred))

    precision_false, precision_true, recall_false, recall_true, fscore_false, fscore_true, support_false, support_true = get_prec_recall_fscore_support(
        y_test, y_pred)

    # precision_false_val, precision_true_val, recall_false_val, recall_true_val, fscore_false_val, fscore_true_val, support_false_val, support_true_val = get_prec_recall_fscore_support(
    #     y_val, y_pred_val)

    if dim_reduc_name is None:
        return acc, precision_false, precision_true, recall_false, recall_true, fscore_false, fscore_true, support_false, support_true

    if hasattr(clf, "hidden_layer_sizes"):
        clf_name = "%s%s" % (clf_name, str(clf.hidden_layer_sizes))

    title = '%s-%s %dD 10FCV\nfold_i=%d, acc=%.1f%%, p0=%d%%, p1=%d%%, r0=%d%%, r1=%d%%\ndataset: class0=%d;' \
            'class1=%d\ntraining: class0=%d; class1=%d\ntesting: class0=%d; class1=%d\nresolution=%s input=%s\n' % (
                clf_name, dim_reduc_name, dim, i,
                acc * 100, precision_false * 100, precision_true * 100, recall_false * 100, recall_true * 100,
                np.count_nonzero(y_lda == 0), np.count_nonzero(y_lda == 1),
                np.count_nonzero(y_train == 0), np.count_nonzero(y_train == 1),
                np.count_nonzero(y_test == 0), np.count_nonzero(y_test == 1), resolution, ','.join(options))

    # title_val = 'VAL %s-%s %dD 10FCV\nfold_i=%d, acc=%.1f%%, p0=%d%%, p1=%d%%, r0=%d%%, r1=%d%%\ndataset: class0=%d;' \
    #         'class1=%d\n' % (
    #     clf_name, dim_reduc_name, dim, i,
    #     acc_val * 100, precision_false_val * 100, precision_true_val * 100, recall_false_val * 100, recall_true_val * 100,
    #     np.count_nonzero(y_val == 0), np.count_nonzero(y_val == 1))

    if dim_reduc_name is '':
        return acc, precision_false, precision_true, recall_false, recall_true, fscore_false, fscore_true, support_false, support_true, \
               title.split('\n')[0], ''

    file_path = None
    if dim == 2 and enalble_2Dplot:
        file_path = plot_2D_decision_boundaries(X_lda, y_lda, X_test, title, clf, folder=folder, options=options, i=i)
        # plot_2D_decision_boundaries(X_val, y_val, X_val, title_val, clf, folder=folder, options=options, i=i)

    if dim == 3 and clf_name is 'SVC' and enalble_3Dplot:
        file_path = plot_3D_decision_boundaries(X_lda, y_lda, X_test, y_test, title, clf, folder=folder,
                                                options=options, i=i)
    if enalble_ROCplot:
        save_roc_curve(y_test, y_probas, title, options, folder, i=i)

    simplified_results = {"accuracy": acc, "specificity": recall_false,
                          "recall": recall_score(y_test, y_pred, average='weighted'),
                          "precision": precision_score(y_test, y_pred, average='weighted'),
                          "f-score": f1_score(y_test, y_pred, average='weighted')}

    return acc, precision_false, precision_true, recall_false, recall_true, fscore_false, fscore_true, support_false, support_true, \
           title.split('\n')[0], file_path, simplified_results


def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = (sum(d[key] for d in dict_list) / len(dict_list)) * 100
    return mean_dict


def process(data_frame, fold=2, dim_reduc=None, clf_name=None, folder=None, options=None, resolution=None):
    X, y, X_val, y_val = process_data_frame(data_frame)
    y = y.astype(int)
    kf = StratifiedKFold(n_splits=fold, random_state=None, shuffle=True)
    kf.get_n_splits(X)

    scores, scores_2d, scores_3d = [], [], []
    precision_false, precision_false_2d, precision_false_3d = [], [], []
    precision_true, precision_true_2d, precision_true_3d = [], [], []
    recall_false, recall_false_2d, recall_false_3d = [], [], []
    recall_true, recall_true_2d, recall_true_3d = [], [], []
    fscore_false, fscore_false_2d, fscore_false_3d = [], [], []
    fscore_true, fscore_true_2d, fscore_true_3d = [], [], []
    support_false, support_false_2d, support_false_3d = [], [], []
    support_true, support_true_2d, support_true_3d = [], [], []
    simplified_results_full, simplified_results_2d, simplified_results_3d = [], [], []

    if clf_name == 'SVC':
        param_grid = {'C': np.logspace(-6, -1, 10), 'gamma': np.logspace(-6, -1, 10)}
        clf = GridSearchCV(SVC(kernel='rbf', probability=True), param_grid, cv=kf)

    if clf_name == 'MLP':
        param_grid = {'hidden_layer_sizes': [(5, 2), (5, 3), (5, 4), (5, 5), (4, 2), (4, 3), (4, 4), (2, 2), (3, 3)],
                      'alpha': [1e-8, 1e-8, 1e-10, 1e-11, 1e-12]}
        clf = GridSearchCV(MLPClassifier(solver='sgd', random_state=1), param_grid, cv=kf)

    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        if dim_reduc == '':
            acc, p_false, p_true, r_false, r_true, fs_false, fs_true, s_false, s_true, clf_name_full, file_path, sr = compute_model(
                X, y, X_val, y_val, train_index, test_index, i, clf=clf, clf_name=clf_name, dim=X.shape[1],
                folder=folder, options=options, resolution=resolution)
            scores.append(acc)
            precision_false.append(p_false)
            precision_true.append(p_true)
            recall_false.append(r_false)
            recall_true.append(r_true)
            fscore_false.append(fs_false)
            fscore_true.append(fs_true)
            support_false.append(s_false)
            support_true.append(s_true)
            simplified_results_full.append(sr)

        if dim_reduc != '':
            acc_2d, p_false_2d, p_true_2d, r_false_2d, r_true_2d, fs_false_2d, fs_true_2d, s_false_2d, s_true_2d, clf_name_2d, file_path_2d, sr_2d = compute_model(
                X, y, X_val, y_val, train_index, test_index, i, clf=clf, dim=2, dim_reduc_name=dim_reduc,
                clf_name=clf_name, folder=folder, options=options, resolution=resolution)
            acc_3d, p_false_3d, p_true_3d, r_false_3d, r_true_3d, fs_false_3d, fs_true_3d, s_false_3d, s_true_3d, clf_name_3d, file_path_3d, sr_3d = compute_model(
                X, y, X_val, y_val, train_index, test_index, i, clf=clf, dim=3, dim_reduc_name=dim_reduc,
                clf_name=clf_name, folder=folder, options=options, resolution=resolution)
            scores_2d.append(acc_2d)
            precision_false_2d.append(p_false_2d)
            precision_true_2d.append(p_true_2d)
            recall_false_2d.append(r_false_2d)
            recall_true_2d.append(r_true_2d)
            fscore_false_2d.append(fs_false_2d)
            fscore_true_2d.append(fs_true_2d)
            support_false_2d.append(s_false_2d)
            support_true_2d.append(s_true_2d)
            simplified_results_2d.append(sr_2d)

            scores_3d.append(acc_3d)
            precision_false_3d.append(p_false_3d)
            precision_true_3d.append(p_true_3d)
            recall_false_3d.append(r_false_3d)
            recall_true_3d.append(r_true_3d)
            fscore_false_3d.append(fs_false_3d)
            fscore_true_3d.append(fs_true_3d)
            support_false_3d.append(s_false_3d)
            support_true_3d.append(s_true_3d)
            simplified_results_3d.append(sr_3d)

    print("svc %d fold cross validation 2d is %f, 3d is %s." % (
        fold, float(np.mean(scores_2d)), float(np.mean(scores_3d))))

    if dim_reduc == '':
        result = {
            'fold': fold,
            'not_reduced': {
                'db_file_path': file_path,
                'clf_name': clf_name_full,
                'scores': scores,
                'accuracy': float(np.mean(scores)),
                'precision_true': float(np.mean(precision_true)),
                'precision_false': np.mean(precision_false),
                'recall_true': float(np.mean(recall_true)),
                'recall_false': np.mean(recall_false),
                'fscore_true': float(np.mean(fscore_true)),
                'fscore_false': float(np.mean(fscore_false)),
                'support_true': np.mean(support_true),
                'support_false': np.mean(support_false)
            },
            'simplified_results': dict_mean(simplified_results_full)
        }
    else:
        result = {
            'fold': fold,
            '2d_reduced': {
                'db_file_path': file_path_2d,
                'clf_name': clf_name_2d,
                'scores': scores_2d,
                'accuracy': float(np.mean(scores_2d)),
                'precision_true': float(np.mean(precision_true_2d)),
                'precision_false': np.mean(precision_false_2d),
                'recall_true': float(np.mean(recall_true_2d)),
                'recall_false': np.mean(recall_false_2d),
                'fscore_true': float(np.mean(fscore_true_2d)),
                'fscore_false': float(np.mean(fscore_false_2d)),
                'support_true': np.mean(support_true_2d),
                'support_false': np.mean(support_false_2d)
            },
            '3d_reduced': {
                'db_file_path': file_path_3d,
                'clf_name': clf_name_3d,
                'scores': scores_3d,
                'accuracy': float(np.mean(scores_3d)),
                'precision_true': float(np.mean(precision_true_3d)),
                'precision_false': np.mean(precision_false_3d),
                'recall_true': float(np.mean(recall_true_3d)),
                'recall_false': np.mean(recall_false_3d),
                'fscore_true': float(np.mean(fscore_true_3d)),
                'fscore_false': float(np.mean(fscore_false_3d)),
                'support_true': np.mean(support_true_3d),
                'support_false': np.mean(support_false_3d)
            },
            'simplified_results': {
                'simplified_results_2d': dict_mean(simplified_results_2d),
                'simplified_results_3d': dict_mean(simplified_results_3d)
            }
        }

    print(result)
    return result


# def classification_report_with_accuracy_score(y_true, y_pred):
#     report = classification_report(y_true, y_pred, output_dict=True)
#     # print(report)
#     precision_true, precision_false, score = parse_report(report)
#     print(precision_true, precision_false)
#     return accuracy_score(y_true, y_pred)
#
#
# def make_meshgrid(x, y, h=.02):
#     x_min, x_max = x.min() - 1, x.max() + 1
#     y_min, y_max = y.min() - 1, y.max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#     return xx, yy
#
#
# def plot_contours(ax, clf, xx, yy, **params):
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     out = ax.contourf(xx, yy, Z, **params)
#     return out
#
#
# def plot_2d_decision_bounderies(Xreduced, y, classifier, title='', folder='', filename=''):
#     fig, ax = plt.subplots()
#     if Xreduced.shape[1] > 1:
#         X0, X1 = Xreduced[:, 0], Xreduced[:, 1]
#         xx, yy = make_meshgrid(X0, X1)
#         plot_contours(ax, classifier, xx, yy, alpha=0.8)
#         ax.scatter(X0, X1, c=y, s=20, edgecolors='k')
#     ax.set_title(title)
#     ax.legend()
#     path = "%s/decision_boundaries_graphs" % folder
#     pathlib.Path(path).mkdir(parents=True, exist_ok=True)
#     final_path = '%s/%s' % (path, filename)
#     fig.savefig(final_path)
#     return final_path


# def get_prec_recall_fscore_support(clf, test_x, test_y, fold):
#     pred_y = cross_val_predict(clf, test_x, test_y, cv=fold)
#     precision_recall_fscore_support_result = precision_recall_fscore_support(test_y, pred_y, average=None,
#                                                                              labels=[False, True])
#     print("precision_recall_fscore_support_result", precision_recall_fscore_support_result)
#     precision_false = precision_recall_fscore_support_result[0][0]
#     precision_true = precision_recall_fscore_support_result[0][1]
#     print("svc precision_false %d fold cross validation is %f" % (fold, precision_recall_fscore_support_result[0][0]))
#     print("svc precision_true %d fold cross validation is %f" % (fold, precision_recall_fscore_support_result[0][1]))
#
#     recall_false = precision_recall_fscore_support_result[1][0]
#     recall_true = precision_recall_fscore_support_result[1][1]
#     print("svc recall_false %d fold cross validation is %f" % (fold, precision_recall_fscore_support_result[1][0]))
#     print("svc recall_true %d fold cross validation is %f" % (fold, precision_recall_fscore_support_result[1][1]))
#
#     fscore_false = precision_recall_fscore_support_result[2][0]
#     fscore_true = precision_recall_fscore_support_result[2][1]
#     print("svc fscore_false %d fold cross validation is %f" % (fold, precision_recall_fscore_support_result[2][0]))
#     print("svc fscore_true %d fold cross validation is %f" % (fold, precision_recall_fscore_support_result[2][1]))
#
#     support_false = precision_recall_fscore_support_result[3][0]
#     support_true = precision_recall_fscore_support_result[3][1]
#     print("svc support_false %d fold cross validation is %f" % (fold, precision_recall_fscore_support_result[3][0]))
#     print("svc support_true %d fold cross validation is %f" % (fold, precision_recall_fscore_support_result[3][1]))
#     return precision_false, precision_true, recall_false, recall_true, fscore_false, fscore_true, support_false, support_true


# def process_SVC_LDA(data_frame, folder, options, clf=SVC(C=100, kernel='rbf'), fold=10):
#     X, y, train_x, _, _, _ = process_data_frame(data_frame)
#     y = y.astype(int)
#     y = np.insert(y, y.size, 3)
#     X = np.vstack((X, np.zeros(X.shape[1])))
#     Xreduced = LDA(n_components=2).fit_transform(X, y)
#     print(Xreduced.shape)
#
#     s = train_x.shape[0]
#     train_x = Xreduced[0:s]
#     train_y = y[0:s]
#     test_x = Xreduced[s:]
#     test_y = y[s:]
#
#     clf = clf.fit(train_x, train_y)
#     scores = cross_val_score(clf, test_x, test_y, cv=fold)
#     print(scores)
#     cross_validated_score = float(np.mean(scores))
#     print("svc %d fold cross validation is %f" % (fold, cross_validated_score))
#     precision_false, precision_true, recall_false, recall_true, fscore_false, fscore_true, support_false, \
#     support_true = get_prec_recall_fscore_support(clf, test_x, test_y, fold)
#
#     timestamp = datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
#     filename = "SVM-rbf-lda_%s_%s.png" % (timestamp, '_'.join(options))
#     db_path = plot_2d_decision_bounderies(Xreduced, y, clf, 'LDA', folder=folder, filename=filename)
#
#     return train_x.shape[0], test_x.shape[0], 1000, 'SVM-rbf-lda', precision_true, precision_false, \
#            cross_validated_score, scores, fold, recall_true, recall_false, \
#            fscore_true, fscore_false, support_true, support_false, X, y, db_path
#
#
# def process_SVC_PCA(data_frame, folder, options, clf=SVC(kernel='rbf', gamma='auto', C=100), fold=10):
#     X, y, train_x, _, _, _ = process_data_frame(data_frame)
#     y = y.astype(int)
#     Xreduced = PCA(n_components=2).fit_transform(X)
#     s = train_x.shape[0]
#     train_x = Xreduced[0:s]
#     train_y = y[0:s]
#     test_x = Xreduced[s:]
#     test_y = y[s:]
#
#     clf.fit(train_x, train_y)
#     scores = cross_val_score(clf, test_x, test_y, cv=fold)
#
#     cross_validated_score = float(np.mean(scores))
#     print("svc %d fold cross validation is %f" % (fold, cross_validated_score))
#
#     precision_false, precision_true, recall_false, recall_true, fscore_false, fscore_true, support_false, \
#     support_true = get_prec_recall_fscore_support(clf, test_x, test_y, fold)
#
#     timestamp = datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
#     filename = "SVM-rbf-pca_%s_%s.png" % (timestamp, '_'.join(options))
#     db_path = plot_2d_decision_bounderies(Xreduced, y, clf, 'PCA', folder=folder, filename=filename)
#
#     return train_x.shape[0], test_x.shape[0], 1000, 'SVM-rbf-pca', precision_true, precision_false, \
#            cross_validated_score, scores, fold, recall_true, recall_false, \
#            fscore_true, fscore_false, support_true, support_false, X, y, db_path
#
#
# def process_SVC(data_frame, clf=SVC(kernel='rbf', gamma='auto', C=1000), fold=10):
#     X, y, train_x, test_x, train_y, test_y = process_data_frame(data_frame)
#     print("Fitting the classifier to the training set")
#     param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 2000, 1000], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1]}
#     clf = GridSearchCV(SVC(gamma='auto', kernel='rbf'), param_grid)
#     clf = clf.fit(train_x, train_y)
#     print("Best estimator found by grid search:")
#     print(clf.best_estimator_)
#     clf = clf.best_estimator_
#     print("predicting...")
#
#     y_pred = clf.predict(test_x)
#     print('Accuracy' + str(accuracy_score(test_y, y_pred)))
#     print(clf.score(test_x, test_y))
#
#     scores = cross_val_score(clf, test_x, test_y, cv=fold)
#     print(scores)
#     cross_validated_score = float(np.mean(scores))
#     print("svc %d fold cross validation is %f" % (fold, cross_validated_score))
#     precision_false, precision_true, recall_false, recall_true, fscore_false, fscore_true, support_false, \
#     support_true = get_prec_recall_fscore_support(clf, test_x, test_y, fold)
#
#     return train_x.shape[0], test_x.shape[0], 1000, 'SVM-rbf', precision_true, precision_false, \
#            cross_validated_score, scores, fold, recall_true, recall_false, \
#            fscore_true, fscore_false, support_true, support_false, X, y, None


# def process_MPL(data_frame, folder,
#                 mpl=MLPClassifier(solver='sgd', alpha=1e-8, hidden_layer_sizes=(5, 2), random_state=1),
#                 fold=10):
#     X, y, train_x, test_x, train_y, test_y = process_data_frame(data_frame)
#     mpl = mpl.fit(train_x, train_y)
#     print("training...")
#     # clf_o.fit(train_x, train_y)
#     print("predicting...")
#
#     y_pred = mpl.predict(test_x)
#     print('Accuracy' + str(accuracy_score(test_y, y_pred)))
#     print(mpl.score(test_x, test_y))
#
#     scores = cross_val_score(mpl, test_x, test_y, cv=fold)
#     print(scores)
#     cross_validated_score = float(np.mean(scores))
#     print("svc %d fold cross validation is %f" % (fold, cross_validated_score))
#     precision_false, precision_true, recall_false, recall_true, fscore_false, fscore_true, support_false, \
#     support_true = get_prec_recall_fscore_support(mpl, test_x, test_y, fold)
#     return train_x.shape[0], test_x.shape[0], 1000, 'NN - sgd (%d,%d)' % (
#         mpl.hidden_layer_sizes[0],
#         mpl.hidden_layer_sizes[1]), precision_true, precision_false, \
#            cross_validated_score, scores, fold, recall_true, recall_false, \
#            fscore_true, fscore_false, support_true, support_false, X, y, None
#
#
# def process_MPL_LDA(data_frame, folder, options,
#                     mpl=MLPClassifier(solver='sgd', alpha=1e-8, hidden_layer_sizes=(5, 2), random_state=1), fold=10):
#     X, y, train_x, _, _, _ = process_data_frame(data_frame)
#     y = y.astype(int)
#     y = np.insert(y, y.size, 3)
#     X = np.vstack((X, np.zeros(X.shape[1])))
#     Xreduced = LDA(n_components=2).fit_transform(X, y)
#     print(Xreduced.shape)
#
#     s = train_x.shape[0]
#     train_x = Xreduced[0:s]
#     train_y = y[0:s]
#     test_x = Xreduced[s:]
#     test_y = y[s:]
#
#     mpl.fit(train_x, train_y)
#     print('reduced', mpl.score(test_x, test_y))
#     scores = cross_val_score(mpl, test_x, test_y, cv=fold)
#     print(scores)
#     cross_validated_score = float(np.mean(scores))
#     print("svc %d fold cross validation is %f" % (fold, cross_validated_score))
#     precision_false, precision_true, recall_false, recall_true, fscore_false, fscore_true, support_false, \
#     support_true = get_prec_recall_fscore_support(mpl, test_x, test_y, fold)
#
#     timestamp = datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
#     filename = "MPL_%s_%s.png" % (timestamp, '_'.join(options))
#     db_path = plot_2d_decision_bounderies(Xreduced, y, mpl, 'LDA', folder=folder, filename=filename)
#
#     return train_x.shape[0], test_x.shape[0], 1000, 'NN - sgd (%d,%d)' % (
#         mpl.hidden_layer_sizes[0],
#         mpl.hidden_layer_sizes[1]), precision_true, precision_false, \
#            cross_validated_score, scores, fold, recall_true, recall_false, \
#            fscore_true, fscore_false, support_true, support_false, X, y, db_path


def process_classifiers(inputs, dir, resolution, dbt, thresh_nan, thresh_zeros):
    filename = init_result_file(dir)
    filename_s = init_result_file(dir, simplified_results=True)
    print("start classification...", inputs)
    start_time = time.time()
    for input in inputs:
        try:
            print(input)
            df = pd.read_csv(input["path"], nrows=1, sep=",", header=None)
            data_col_n = df.iloc[[0]].size
            type_dict = {}
            for n, i in enumerate(range(0, data_col_n)):
                if n < (data_col_n - 5):
                    type_dict[str(i)] = np.float16
                else:
                    type_dict[str(i)] = np.str

            # data_frame = pd.read_csv(input["path"], sep=",", header=None, dtype=type_dict, low_memory=False)
            data_frame = pd.read_csv(input["path"], sep=",", header=None, dtype=type_dict, low_memory=False)
            # data_frame = pd.DataFrame()
            # chunk_size = 10000
            # i = 1
            # for chunk in pd.read_csv(input["path"], chunksize=chunk_size, low_memory=False):
            #     data_frame = chunk if i == 1 else pd.concat([df, chunk])
            #     print('-->reading chunck...', i)
            #     i += 1

            sample_count = data_frame.shape[1]
            header = [str(n) for n in range(0, sample_count)]
            header[-5] = "class"
            header[-4] = "elem_in_row"
            header[-3] = "date1"
            header[-2] = "date2"
            header[-1] = "serial"
            data_frame.columns = header
            data_frame = data_frame.loc[:, :'class']
            np.random.seed(0)
            data_frame = data_frame.sample(frac=1).reset_index(drop=True)
            data_frame = data_frame.fillna(-1)
            data_frame = shuffle(data_frame)
            print(data_frame)

            class_true_count = data_frame['class'].value_counts().to_dict()[True]
            class_false_count = data_frame['class'].value_counts().to_dict()[False]
            print("class_true_count=%d and class_false_count=%d" % (class_true_count, class_false_count))

            for result in [
                process(data_frame, fold=10, dim_reduc='LDA', clf_name='SVC', folder=dir,
                        options=input["options"], resolution=resolution)
                # process(data_frame, fold=10, dim_reduc='LDA', clf_name='MLP', folder=dir,
                #         options=input["options"], resolution=resolution)
                # process(data_frame, fold=10, dim_reduc='', clf_name='SVC', folder=dir,
                #         options=input["options"], resolution=resolution),
            ]:
                time_proc = get_elapsed_time_string(start_time, time.time())

                if '2d_reduced' in result:
                    r_2d = result['2d_reduced']
                    append_result_file(filename, r_2d['accuracy'], r_2d['scores'], r_2d['precision_true'],
                                       r_2d['precision_false'],
                                       r_2d['recall_true'], r_2d['recall_false'], r_2d['fscore_true'],
                                       r_2d['fscore_false'], r_2d['support_true'], r_2d['support_false'],
                                       class_true_count,
                                       class_false_count, result['fold'],
                                       resolution, dbt, thresh_nan, thresh_zeros, time_proc, sample_count,
                                       data_frame.shape[0],
                                       input["path"], input["options"], r_2d['clf_name'], r_2d['db_file_path'])

                    r_3d = result['3d_reduced']
                    append_result_file(filename, r_3d['accuracy'], r_3d['scores'], r_3d['precision_true'],
                                       r_3d['precision_false'],
                                       r_3d['recall_true'], r_3d['recall_false'], r_3d['fscore_true'],
                                       r_3d['fscore_false'], r_3d['support_true'], r_3d['support_false'],
                                       class_true_count,
                                       class_false_count, result['fold'],
                                       resolution, dbt, thresh_nan, thresh_zeros, time_proc, sample_count,
                                       data_frame.shape[0],
                                       input["path"], input["options"], r_3d['clf_name'], r_3d['db_file_path'])

                if 'not_reduced' in result:
                    r = result['not_reduced']
                    append_result_file(filename, r['accuracy'], r['scores'], r['precision_true'], r['precision_false'],
                                       r['recall_true'], r['recall_false'], r['fscore_true'],
                                       r['fscore_false'], r['support_true'], r['support_false'], class_true_count,
                                       class_false_count, result['fold'],
                                       resolution, dbt, thresh_nan, thresh_zeros, time_proc, sample_count,
                                       data_frame.shape[0],
                                       input["path"], input["options"], r['clf_name'], r['db_file_path'])

                if 'simplified_results' in result:
                    if 'simplified_results_2d' in result['simplified_results']:
                        item = result['simplified_results']['simplified_results_2d']
                        append_simplified_result_file(filename_s, r_2d['clf_name'], item['accuracy'],
                                                      item['specificity'],
                                                      item['recall'], item['precision'], item['f-score'],
                                                      days_before_famacha_test, resolution,
                                                      format_options(input["options"]))
                    if 'simplified_results_3d' in result['simplified_results']:
                        item = result['simplified_results']['simplified_results_3d']
                        append_simplified_result_file(filename_s, r_3d['clf_name'], item['accuracy'],
                                                      item['specificity'],
                                                      item['recall'], item['precision'], item['f-score'],
                                                      days_before_famacha_test, resolution,
                                                      format_options(input["options"]))
                    if 'simplified_results_full' in result['simplified_results']:
                        item = result['simplified_results']['simplified_results_full']
                        append_simplified_result_file(filename_s, r['clf_name'], item['accuracy'], item['specificity'],
                                                      item['recall'], item['precision'], item['f-score'],
                                                      days_before_famacha_test, resolution,
                                                      format_options(input["options"]))

            # for train_x_size, test_x_size, penalty, classifier, precision_true, precision_false, cross_validated_score, scores, fold, \
            #     recall_true, recall_false, fscore_true, fscore_false, support_true, support_false, X, y, db_path in [
            #     process_SVC_LDA(data_frame, dir, input["options"]), process_SVC(data_frame),
            #     process_SVC_PCA(data_frame, dir, input["options"])]:
            #     time_proc = get_elapsed_time_string(start_time, time.time())
            #
            #     append_result_file(filename, cross_validated_score, scores, precision_true, precision_false,
            #                        recall_true, recall_false, fscore_true, fscore_false, support_true,
            #                        support_false, class_true_count,
            #                        class_false_count, fold, resolution, dbt, thresh_nan, thresh_zeros, time_proc,
            #                        sample_count,
            #                        data_frame.shape[0],
            #                        input["path"], input["options"], classifier, db_path)
        except ValueError as e:
            #LDA(n_components=output_dim).fit_transform(X_train, y_train) usually culprit
            print(e)
            continue


def format_options(options):
    return '+'.join(options).replace('humidity', 'h').replace('temperature', 't').replace('activity', 'a').replace(
        'indexes', 'i')


def merge_results(filename="results_report_%s.xlsx" % run_timestamp, filter='results.csv'):
    purge_file(filename)
    directory_path = os.getcwd().replace('C','E')
    os.chdir(directory_path)
    file_paths = [val for sublist in
                  [[os.path.join(i[0], j) for j in i[2] if j.endswith(filter)] for i in os.walk(directory_path)]
                  for
                  val in sublist]

    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0
    # Iterate over the data and write it out row by row.
    for item in RESULT_FILE_HEADER.split(','):
        worksheet.write(row, col, item)
        col += 1
    row = 1
    for file in file_paths:
        for idx, line in enumerate(open(file)):
            if idx == 0:
                continue
            col = 0
            for idx, item in enumerate(line.split(',')):
                if idx == 28:
                    if 'None' in item:
                        continue
                    item = item.replace('\\', '/').replace('\n', '')
                    worksheet.write(row, col, item)
                    worksheet.insert_image(row=row, col=col + 1, filename=item,
                                           options={'x_scale': 0.05, 'y_scale': 0.05})
                    continue
                worksheet.write(row, col, item)
                col += 1
            row += 1
    workbook.close()

    # with open(filename, 'a') as outfile:
    #     outfile.write(RESULT_FILE_HEADER)
    #     outfile.write('\n')
    #
    # for file in file_paths:
    #     for idx, line in enumerate(open(file)):
    #         if idx == 0:
    #             continue
    #         with open(filename, 'a') as outfile:
    #             outfile.write(line)
    #
    # outfile.close()
    exit(0)


if __name__ == '__main__':
    print("pandas", pd.__version__)
    farm_id = "bothaville_70091100060"
    resolution_l = ['10min', '5min', 'hour', 'day']
    days_before_famacha_test_l = [2]
    threshold_nan_coef = 5
    threshold_zeros_coef = 2
    nan_threshold, zeros_threshold = 0, 0
    start_time = time.time()
    print('args=', sys.argv)
    connect_to_sql_database()
    create_cwt_graph_enabled = False
    create_activity_graph_enabled = True

    for resolution in resolution_l:
        if resolution == "min":
            threshold_nan_coef = 1.5
            threshold_zeros_coef = 1.5
        # if resolution == "day":
        #     days_before_famacha_test_l = [3, 4, 5, 6]

        for days_before_famacha_test in days_before_famacha_test_l:
            expected_sample_count = get_expected_sample_count(resolution, days_before_famacha_test)

            # generate_training_sets(data_famacha_flattened)
            with open(os.path.join(__location__, 'delmas_weather.json')) as f:
                weather_data = json.load(f)



            #data_famacha_dict = generate_table_from_xlsx('Lange-Henry-Debbie-Skaap-Jun-2016a.xlsx')

            with open('C:\\Users\\fo18103\\PycharmProjects\\prediction_of_helminths_infection\\db_processor\\src\\bothaville_famacha_data.json', 'r') as fp:
                data_famacha_dict = json.load(fp)
                print(data_famacha_dict.keys())

            data_famacha_list = [y for x in data_famacha_dict.values() for y in x]
            results = []
            herd_data = []
            dir = "%s/%s_resolution_%s_days_%d_log" % (os.getcwd().replace('C','E'), farm_id, resolution, days_before_famacha_test)
            class_input_dict_file_path = dir + '/class_input_dict.json'
            if False:#os.path.exists(class_input_dict_file_path):
                print('training sets already created skip to processing.')
                with open(class_input_dict_file_path, "r") as read_file:
                    class_input_dict = json.load(read_file)
            else:
                print('start training sets creation...')
                try:
                    shutil.rmtree(dir)
                except (OSError, FileNotFoundError) as e:
                    print(e)
                    # exit(-1)

                for curr_data_famacha in data_famacha_list:
                    try:
                        result = get_training_data(curr_data_famacha, data_famacha_dict, weather_data, resolution,
                                               days_before_famacha_test, expected_sample_count, farm_sql_table_id=farm_id)
                    except KeyError as e:
                        print(e)

                    if result is None:
                        continue

                    is_valid, nan_threshold, zeros_threshold = is_activity_data_valid(result["activity"], threshold_nan_coef,
                                                                                      threshold_zeros_coef)
                    result['is_valid'] = True
                    if not is_valid:
                        result['is_valid'] = False

                    result["nan_threshold"] = nan_threshold
                    result["zeros_threshold"] = zeros_threshold
                    results.append(result)

                skipped_class_false, skipped_class_true = process_famacha_var(results)

                class_input_dict = []
                for result in results:
                    if not result['is_valid']:
                        continue
                    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
                    filename = create_filename(result)
                    if create_activity_graph_enabled:
                        create_activity_graph(result["activity"], dir, filename, title=create_graph_title(result, "time"))

                    # cwt, coef, freqs, indexes_cwt = compute_cwt(result["activity"])

                    cwt, coef, freqs, indexes_cwt, scales, delta_t, wavelet_type = compute_hd_cwt(result["activity"])

                    # weight = process_weight(result["activity"], coef)
                    result["cwt"] = cwt
                    result["coef_shape"] = coef.shape
                    # result["cwt_weight"] = weight
                    result["indexes_cwt"] = indexes_cwt

                    herd_data.append(result['herd'])
                    if create_cwt_graph_enabled:
                        create_hd_cwt_graph(coef, dir, filename, title=create_graph_title(result, "freq"))

                    class_input_dict = create_training_sets(result, dir)  # warning! always returns the same result
                    if not os.path.exists(class_input_dict_file_path):
                        with open(class_input_dict_file_path, 'w') as fout:
                            json.dump(class_input_dict, fout)

            herd_file_path = dir + '/%s_herd_activity.json' % farm_id
            if not os.path.exists(herd_file_path):
                with open(herd_file_path, 'w') as fout:
                    json.dump({'herd_activity': herd_data}, fout)

            exit()
            process_classifiers(class_input_dict, dir, resolution, days_before_famacha_test, nan_threshold,
                                zeros_threshold)

    print(get_elapsed_time_string(start_time, time.time()))
    merge_results(filename="results_simplified_report_%s.xlsx" % run_timestamp, filter='results_simplified.csv')
    merge_results(filename="results_report_%s.xlsx" % run_timestamp, filter='results.csv')
