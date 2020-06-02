import base64
import json
import json
import math
import gc
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
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors.classification import KNeighborsClassifier
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
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold
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
import multiprocessing
from multiprocessing import Pool, freeze_support
import itertools
from socket import *
from functools import partial
from sys import exit
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import auc
from scipy import interp
from matplotlib.colors import LinearSegmentedColormap

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
                     "fold,proba_y_false, proba_y_true,resolution," \
                     "days_before_test,sliding_w,threshold_nan,threshold_zeros,processing_time,sample_count,set_size," \
                     "file_path,input,classifier, decision_bounderies_file"

RESULT_FILE_HEADER_SIMPLIFIED = "classifier, accuracy,specificity,recall,precision,fscore,proba_y_false,proba_y_true,days,sliding_w,resolution,inputs"

skipped_class_false, skipped_class_true = -1, -1
META_DATA_LENGTH = 19


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NonDaemonicPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


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


def get_weight(curr_datetime, data_famacha_dict, animal_id):
    weight = None
    c = curr_datetime.strftime("%d/%m/%Y")
    for data in data_famacha_dict[str(animal_id)]:
        if c in data[0]:
            try:
                weight = float(data[3])
            except ValueError as e:
                print(e)
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
        print("could not find weather data!", e)

    return temp, humidity


def get_prev_famacha_score(serial_number, famacha_test_date, data_famacha, curr_score):
    previous_score1 = None
    previous_score2 = None
    previous_score3 = None
    previous_score4 = None
    try:
        list = data_famacha[str(serial_number)]
    except KeyError as e:
        print(e)
        exit()
    for i in range(1, len(list)):
        item = list[i]
        if item[0] == famacha_test_date:
            try:
                previous_score1 = int(list[i - 1][1])
            except ValueError as e:
                previous_score1 = -1
                print(e)
            try:
                previous_score2 = int(list[i - 2][1])
            except ValueError as e:
                previous_score2 = -1
                print(e)
            try:
                previous_score3 = int(list[i - 3][1])
            except ValueError as e:
                previous_score3 = -1
                print(e)
            try:
                previous_score4 = int(list[i - 4][1])
            except ValueError as e:
                previous_score4 = -1
                print(e)
            break
    return previous_score1, previous_score2, previous_score3, previous_score4


def pad(a, N):
    a += [-1] * (N - len(a))
    return a


def connect_to_sql_database(db_server_name="localhost", db_user="axel", db_password="Mojjo@2015",
                            db_name="south_africa_debug",
                            char_set="utf8mb4", cusror_type=pymysql.cursors.DictCursor):
    # print("connecting to db %s..." % db_name)
    sql_db = pymysql.connect(host=db_server_name, user=db_user, password=db_password,
                             db=db_name, charset=char_set, cursorclass=cusror_type)
    return sql_db


def execute_sql_query(sql_db, query, records=None, log_enabled=False):
    try:
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
    return 2 * math.sqrt(abs(value) + (3 / 8))

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


def normalize_histogram_mean_diff(activity_mean, activity):
    scale = [0 for _ in range(0, len(activity))]
    idx = []
    for n, a in enumerate(activity):
        if a is None or a <= 0:
            continue
        if activity_mean[n] is None:
            continue
        r = (int(activity_mean[n]) / int(a))

        scale[n] = r
        idx.append(n)
    median = math.fabs(statistics.median(sorted(set(scale))))
    #print(scale)
    for i in idx:
        activity[i] = activity[i] * median
    return activity


def get_period(curr_data_famacha, days_before_famacha_test, sliding_window):
    famacha_test_date = time.strptime(curr_data_famacha[0], "%d/%m/%Y")
    famacha_test_date_epoch_s = str(time.mktime((datetime.fromtimestamp(time.mktime(famacha_test_date)) -
                                                        timedelta(days=sliding_window)).timetuple())).split('.')[0]
    famacha_test_date_epoch_before_s = str(time.mktime((datetime.fromtimestamp(time.mktime(famacha_test_date)) -
                                                        timedelta(days=sliding_window + days_before_famacha_test)).
                                                       timetuple())).split('.')[0]
    famacha_test_date_formated = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(famacha_test_date_epoch_s)))
    famacha_test_date_epoch_before_formated = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(famacha_test_date_epoch_before_s)))
    return famacha_test_date_epoch_s, famacha_test_date_epoch_before_s, famacha_test_date_formated, famacha_test_date_epoch_before_formated


def sql_dict_list_to_list(dict_list):
    return pd.DataFrame(dict_list)['serial_number'].tolist()


def format_cedara_famacha_data(data_famacha_dict, sql_db):
    # famacha records contains shorten version of the serial numbers whereas activity records contains full version
    rows_serial_numbers = execute_sql_query(sql_db,
        "SELECT DISTINCT serial_number FROM cedara_70091100056_resolution_10min")
    serial_numbers_full = sql_dict_list_to_list(rows_serial_numbers)

    data_famacha_dict_formatted = {}
    for key, value in data_famacha_dict.items():
        for elem in serial_numbers_full:
            if key in str(elem):
                v = value.copy()
                for item in v:
                    item[2] = elem
                data_famacha_dict_formatted[str(elem)] = v

    return data_famacha_dict_formatted


def get_ndays_between_dates(date1, date2):
    date_format = "%d/%m/%Y"
    a = datetime.strptime(date1, date_format)
    b = datetime.strptime(date2, date_format)
    delta = b - a
    return delta.days


def get_training_data(sql_db, curr_data_famacha, i, data_famacha_list, data_famacha_dict, weather_data, resolution, days_before_famacha_test,
                      expected_sample_count, farm_sql_table_id=None, sliding_windows=None):
    # print("generating new training pair....")
    famacha_test_date = datetime.fromtimestamp(time.mktime(time.strptime(curr_data_famacha[0], "%d/%m/%Y"))).strftime(
        "%d/%m/%Y")
    try:
        famacha_score = int(curr_data_famacha[1])
    except ValueError as e:
        print("error while parsing famacha score!", e)
        return

    animal_id = int(curr_data_famacha[2])
    # find the activity data of that animal the n days before the test
    date1, date2, _, _ = get_period(curr_data_famacha, days_before_famacha_test, sliding_windows)

    dtf1, dtf2, dtf3, dtf4, dtf5 = "", "", "", "", ""
    try:
        dtf1 = data_famacha_list[i][0]
    except IndexError as e:
        print(e)
    try:
        dtf2 = data_famacha_list[i+1][0]
    except IndexError as e:
        print(e)
    try:
        dtf3 = data_famacha_list[i+2][0]
    except IndexError as e:
        print(e)
    try:
        dtf4 = data_famacha_list[i+3][0]
    except IndexError as e:
        print(e)
    try:
        dtf5 = data_famacha_list[i+4][0]
    except IndexError as e:
        print(e)

    nd1, nd2, nd3, nd4 = 0, 0, 0, 0
    if len(dtf2) > 0 and len(dtf1) > 0:
        nd1 = abs(get_ndays_between_dates(dtf1, dtf2))
    if len(dtf3) > 0 and len(dtf2) > 0:
        nd2 = abs(get_ndays_between_dates(dtf2, dtf3))
    if len(dtf4) > 0 and len(dtf3) > 0:
        nd3 = abs(get_ndays_between_dates(dtf3, dtf4))
    if len(dtf5) > 0 and len(dtf4) > 0:
        nd4 = abs(get_ndays_between_dates(dtf4, dtf5))


    print("%s getting activity data for test on the %s for %d. collecting data %d days before resolution is %s..." % (
        farm_sql_table_id, famacha_test_date, animal_id, days_before_famacha_test, resolution))

    rows_activity = execute_sql_query(sql_db, "SELECT timestamp, serial_number, first_sensor_value FROM %s_resolution_%s"
                                      " WHERE timestamp BETWEEN %s AND %s AND serial_number = %s" %
                                      (farm_sql_table_id, resolution, date2, date1,
                                       str(animal_id)))

    rows_herd = execute_sql_query(sql_db,
        "SELECT timestamp, serial_number, first_sensor_value FROM %s_resolution_%s WHERE serial_number=%s AND timestamp BETWEEN %s AND %s" %
        (farm_sql_table_id, resolution, 50000000000, date2, date1))

    # activity_mean = normalize_activity_array_ascomb(rows_mean)
    herd_activity_list = [(x['first_sensor_value']) for x in rows_herd]
    activity_list = [(x['first_sensor_value']) for x in rows_activity]

    if len(rows_activity) != expected_sample_count:
        # filter out missing activity data
        print("absent activity records. skip.", "found %d" % len(rows_activity), "expected %d" % expected_sample_count)
        return

    # data_activity = normalize_activity_array_ascomb(data_activity)
    herd_activity_list = anscombe_list(herd_activity_list)
    activity_list = anscombe_list(activity_list)

    # herd_activity_list = anscombe_list(herd_activity_list)
    # activity_list = anscombe_list(activity_list)

    activity_list = normalize_histogram_mean_diff(herd_activity_list, activity_list)

    # print("mapping activity to famacha score progress=%d/%d ..." % (i, len(data_famacha_flattened)))
    idx = 0
    indexes = []
    timestamp_list = []
    humidity_list = []
    weight_list = []
    temperature_list = []
    dates_list_formated = []

    for j, data_a in enumerate(rows_activity):
        # transform date in time for comparaison
        curr_datetime = datetime.utcfromtimestamp(int(data_a['timestamp']))
        timestamp = time.strptime(curr_datetime.strftime('%d/%m/%Y'), "%d/%m/%Y")
        temp, humidity = get_temp_humidity(curr_datetime, weather_data)

        try:
            weight = float(curr_data_famacha[3])
        except ValueError as e:
            print(e)
            weight = None

        weight_list.append(weight)

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

    prev_famacha_score1, prev_famacha_score2, prev_famacha_score3, prev_famacha_score4 = get_prev_famacha_score(animal_id,
                                                                                           famacha_test_date,
                                                                                           data_famacha_dict,
                                                                                           famacha_score)
    indexes.reverse()

    # herd_activity_list = anscombe_list(herd_activity_list)
    # activity_list = anscombe_list(activity_list)

    data = {"famacha_score_increase": False, "famacha_score": famacha_score, "weight": weight_list,
            "previous_famacha_score1": prev_famacha_score1,
            "previous_famacha_score2": prev_famacha_score2,
            "previous_famacha_score3": prev_famacha_score3,
            "previous_famacha_score4": prev_famacha_score4,
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
            "indexes": indexes, "activity": activity_list,
            "temperature": temperature_list, "humidity": humidity_list, "herd": herd_activity_list, "ignore": True
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


def entropy2(labels, base=None):
  """ Computes entropy of label distribution. """
  n_labels = len(labels)
  if n_labels <= 1:
    return 0
  value,counts = np.unique(labels, return_counts=True)
  probs = counts / n_labels
  n_classes = np.count_nonzero(probs)
  if n_classes <= 1:
    return 0
  ent = 0.
  # Compute entropy
  base = math.e if base is None else base
  for i in probs:
    ent -= i * math.log(i, base)
  return ent


def is_activity_data_valid(activity, threshold_nan_coef, threshold_zeros_coef):

    # nan_threshold, zeros_threshold = 0, 0
    nan_threshold = len(activity) / threshold_nan_coef
    zeros_threshold = len(activity) / threshold_zeros_coef


    activity_np = np.asarray(activity, dtype=np.float)
    np.nan_to_num(activity_np, nan=-1)
    a, b = np.unique(activity_np, return_counts=True)
    most_abundant_value = a[b.argmax()]
    occurance = b.max()
    print(most_abundant_value, occurance)
    if most_abundant_value == np.nan or occurance > 2500 or most_abundant_value > 500:
        return False, nan_threshold, zeros_threshold, 0

    nan_count = activity.count(None)
    zeros_count = activity.count(0)
    # print(nan_count, zeros_count, nan_threshold, zeros_threshold)
    if nan_count > int(nan_threshold/1) or zeros_count > zeros_threshold :#or contains_negative(activity):
        return False, nan_threshold, zeros_threshold, 0

    # plt.plot(activity_np)
    # plt.show()

    h = entropy2(activity_np)
    print(h)
    ENTROPY_THRESH = 3.5
    if h <= ENTROPY_THRESH:
        return False, nan_threshold, zeros_threshold, h

    return True, nan_threshold, zeros_threshold, h


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


def create_activity_graph(activity, folder, filename, title=None, sub_folder='training_sets_time_domain_graphs', sub_sub_folder=None):
    activity = [-20 if x is None else x for x in activity]
    fig = plt.figure()
    plt.bar(range(0, len(activity)), activity)
    fig.suptitle(title, x=0.5, y=.95, horizontalalignment='center', verticalalignment='top', fontsize=10)
    path = "%s/%s/%s" % (folder, sub_folder, sub_sub_folder)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    fig.savefig('%s/%s' % (path, filename))
    fig.clear()
    plt.close(fig)


def create_cwt_graph(coef, freqs, lenght, folder, filename, title=None):
    time = [x for x in range(0, lenght)]
    fig = plt.figure()
    plt.pcolormesh(time, freqs, coef)
    fig.suptitle(title, x=0.5, y=.95, horizontalalignment='center', verticalalignment='top', fontsize=10)
    path = "%s/training_sets_cwt_graphs" % folder
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    fig.savefig('%s/%s' % (path, filename))
    coef_f = coef.flatten().tolist()


def create_hd_cwt_graph(coefs, folder, filename, title=None, sub_folder='training_sets_cwt_graphs', sub_sub_folder=None, freqs=None):
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
    fig.savefig('%s/%s' % (path, filename))
    fig.clear()
    plt.close(fig)

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
    wavelet_type = 'gaus8'
    w = pywt.ContinuousWavelet(wavelet_type)
    scales = even_list(40)
    sampling_frequency = 1 / 60
    sampling_period = 1 / sampling_frequency
    activity_i = interpolate(activity)
    coef, freqs = pywt.cwt(np.asarray(activity_i), scales, w, sampling_period=sampling_period)
    cwt = [element for tupl in coef for element in tupl]
    # indexes = list(range(len(cwt)))
    # indexes.reverse()
    indexes = []
    return cwt, coef, freqs, indexes, scales, 1, wavelet_type


def create_filename(data):
    filename = "%.2f_famacha[%.1f]_%d_%s_%s_sd_pvfs%d_zt%d_nt%d.png" % (data['entropy'], data["famacha_score"], data["animal_id"], data["date_range"][0], data["date_range"][1],
                                                       -1 if data["previous_famacha_score1"] is None else data["previous_famacha_score1"],
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
    filename = "%s/%s.data" % (path, option)
    training_str_flatten = str(training_set).strip('[]').replace(' ', '').replace('None', 'NaN')
    print("set size is %d, %s.....%s" % (
        len(training_set), training_str_flatten[0:50], training_str_flatten[-50:]))
    with open(filename, 'a') as outfile:
        outfile.write(training_str_flatten)
        outfile.write('\n')

    filename_t = "%s/temperature.data" % path
    temp_str_flatten = str(result["temperature"]).strip('[]').replace(' ', '').replace('None', 'NaN')
    print("set size is %d, %s.....%s" % (
        len(temp_str_flatten), temp_str_flatten[0:50], temp_str_flatten[-50:]))
    with open(filename_t, 'a') as outfile_t:
        outfile_t.write(temp_str_flatten)
        outfile_t.write('\n')

    filename_h = "%s/humidity.data" % path
    hum_str_flatten = str(result["humidity"]).strip('[]').replace(' ', '').replace('None', 'NaN')
    print("set size is %d, %s.....%s" % (
        len(hum_str_flatten), hum_str_flatten[0:50], hum_str_flatten[-50:]))
    with open(filename_h, 'a') as outfile_h:
        outfile_h.write(hum_str_flatten)
        outfile_h.write('\n')

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
    path1, options1 = create_training_set(data, dir_path, options=["activity"])
    path2, options2 = create_training_set(data, dir_path, options=["cwt"])
    # path3, options3 = create_training_set(data, dir_path, options=["weight"])
    # path4, options4 = create_training_set(data, dir_path, options=["activity", "temperature"])
    # path5, options5 = create_training_set(data, dir_path, options=["activity", "humidity"])
    path6, options6 = create_training_set(data, dir_path, options=["activity", "weight"])
    path7, options7 = create_training_set(data, dir_path, options=["activity", "humidity", "temperature"])
    # path8, options8 = create_training_set(data, dir_path, options=["activity", "humidity", "temperature", "weight"])
    # path9, options9 = create_training_set(data, dir_path, options=["cwt", "humidity"])
    # path10, options10 = create_training_set(data, dir_path, options=["cwt", "temperature"])
    path11, options11 = create_training_set(data, dir_path, options=["cwt", "weight"])
    path12, options12 = create_training_set(data, dir_path, options=["cwt", "humidity", "temperature"])
    # path13, options13 = create_training_set(data, dir_path, options=["cwt", "humidity", "temperature", "weight"])

    return [
        {"path": path1, "options": options1},
            {"path": path2, "options": options2},
            # {"path": path3, "options": options3},
            # {"path": path4, "options": options4},
            # {"path": path5, "options": options5},
            {"path": path6, "options": options6},
            {"path": path7, "options": options7},
            # {"path": path8, "options": options8},
            # {"path": path9, "options": options9},
            # {"path": path10, "options": options10},
            {"path": path11, "options": options11},
            {"path": path12, "options": options12}
            # {"path": path13, "options": options13}
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


def init_result_file(dir, farm_id, simplified_results=False):
    path = "%s/analysis" % dir
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    filename = "%s/%s_results_simplified.csv" % (path, farm_id) if simplified_results else "%s/%s_results.csv" % (path, farm_id)
    with open(filename, 'a') as outfile:
        outfile.write(RESULT_FILE_HEADER_SIMPLIFIED) if simplified_results else outfile.write(RESULT_FILE_HEADER)
        outfile.write('\n')
    outfile.close()
    return filename


def append_simplified_result_file(filename, classifier_name, accuracy, specificity, recall, precision, fscore,
                                  proba_y_false, proba_y_true,
                                  days_before_test, sliding_w, resolution, options):
    data = "%s, %.2f,%.2f,%.2f,%.2f,%.2f,%s,%s,%d,%d,%s,%s" % (
    classifier_name.replace(',', ':').replace(' 10FCV', ''), accuracy, specificity, recall, precision, fscore,
    str(proba_y_false).replace(',', '-'), str(proba_y_true).replace(',', '-'),
    days_before_test, sliding_w, resolution, options)
    with open(filename, 'a') as outfile:
        outfile.write(data)
        outfile.write('\n')
    outfile.close()


def append_result_file(filename, cross_validated_score, scores, precision_true, precision_false,
                       recall_true, recall_false, fscore_true, fscore_false, support_true, support_false,
                       class_true_count, class_false_count, fold,
                       proba_y_false, proba_y_true,
                       resolution, days_before_test, sliding_w, threshold_nan, threshold_zeros,
                       processing_time,
                       sample_count, set_size, training_file, options, kernel,
                       db_path):
    global skipped_class_false
    global skipped_class_true
    scores_s = ' '.join([str(x) for x in scores])
    precision_true = ' '.join([str(x) for x in precision_true])
    precision_false = ' '.join([str(x) for x in precision_false])
    recall_true = ' '.join([str(x) for x in recall_true])
    recall_false = ' '.join([str(x) for x in recall_false])
    fscore_true = ' '.join([str(x) for x in fscore_true])
    fscore_false = ' '.join([str(x) for x in fscore_false])
    proba_y_false = ' '.join([str(x) for x in proba_y_false])
    proba_y_true = ' '.join([str(x) for x in proba_y_true])
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
    print('proba_y_false', type(proba_y_false), proba_y_false)
    print('proba_y_true', type(proba_y_true), proba_y_true)
    print('resolution', type(resolution), resolution)
    print('days_before_test', type(days_before_test), days_before_test)
    print('sliding_w', type(sliding_w), sliding_w)
    print('threshold_nan', type(threshold_nan), threshold_nan)
    print('threshold_zeros', type(threshold_zeros), threshold_zeros)
    print('processing_time', type(processing_time), processing_time)
    print('sample_count', type(sample_count), sample_count)
    print('set_size', type(set_size), set_size)
    print('training_file', type(training_file), training_file)
    print('options', type(options), options)
    print('kernel', type(kernel), kernel)
    print('db_path', type(db_path), db_path)
    data = "%.15f,%s,%s,%s,%s,%s,%s,%s,%d,%d,%d,%d,%d,%d,%d,%s,%s,%s,%d,%d,%d,%d,%s,%d,%d,%s,%s,%s,%s" % (
        cross_validated_score, scores_s, precision_true, precision_false, recall_true, recall_false, fscore_true,
        fscore_false,
        support_true, support_false, class_true_count, class_false_count, skipped_class_true, skipped_class_false, fold,
        proba_y_false, proba_y_true,
        resolution, days_before_test, sliding_w, threshold_nan, threshold_zeros,
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


def process_data_frame(data_frame, y_col='label'):
    data_frame = data_frame.fillna(-1)
    cwt_shape = data_frame[data_frame.columns[0:2]].values
    X = data_frame[data_frame.columns[2:data_frame.shape[1] - META_DATA_LENGTH]].values
    X = normalize(X)
    X = preprocessing.MinMaxScaler().fit_transform(X)
    y = data_frame[y_col].values.flatten()
    y = y.astype(int)
    return X, y

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
    sub_folder = "%s" % (title.split('\n')[0])
    sub_folder = sub_folder.replace(',', '-').replace(' ', '').replace(' ', '_').replace('-', '_')
    sub_folder = '_'.join(options) + '_' + '_'.join(sub_folder.split('_')[0:2])
    return format_options_(sub_folder)


def get_conf_interval(tprs, mean_fpr):
    confidence_lower = []
    confidence_upper = []
    df_tprs = pd.DataFrame(tprs, dtype=float)
    for column in df_tprs:
        scores = df_tprs[column].values.tolist()
        scores.sort()
        upper = np.percentile(scores, 95)
        confidence_upper.append(upper)
        lower = np.percentile(scores, 0.025)
        confidence_lower.append(lower)

    confidence_lower = np.asarray(confidence_lower)
    confidence_upper = np.asarray(confidence_upper)
    # confidence_upper = np.minimum(mean_tpr + std_tpr, 1)
    # confidence_lower = np.maximum(mean_tpr - std_tpr, 0)

    return confidence_lower, confidence_upper


def plot_roc_range(ax, tprs, mean_fpr, aucs, fig, title, options, folder, i=0):
    print("plot_roc_range...")
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='orange',
            label='Chance', alpha=1)

    mean_tpr = np.mean(tprs, axis=0)
    # mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='tab:blue',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    confidence_lower, confidence_upper = get_conf_interval(tprs, mean_fpr)

    ax.fill_between(mean_fpr, confidence_lower, confidence_upper, color='tab:blue', alpha=.2)
                    #label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="Receiver operating characteristic")
    ax.legend(loc="lower right")
    # fig.show()
    path = "%s/roc_curve/%s" % (folder, format_sub_folder_name(title, options))
    print(path)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    final_path = '%s/%s' % (path, format_file_name(i, title, options))
    final_path = final_path.replace('/', '\'').replace('\'', '\\').replace('\\', '/')
    print(final_path)
    fig.savefig(final_path)


def plot_2D_decision_boundaries(X_lda, y_lda, X_test, y_test, title, clf, folder=None, i=0, options=None, n_bin=8):
    print('graph...')
    # plt.subplots_adjust(top=0.75)
    # fig = plt.figure(figsize=(7, 6), dpi=100)
    fig, ax = plt.subplots(figsize=(7., 4.8))
    # plt.subplots_adjust(top=0.75)
    min = abs(X_lda.min()) + 1
    max = abs(X_lda.max()) + 1
    print(X_lda.shape)
    print(min, max)
    if np.max([min, max]) > 100:
        return
    xx, yy = np.mgrid[-min:max:.01, -min:max:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = clf.predict_proba(grid)[:, 1].reshape(xx.shape)
    offset_r = 0
    offset_g = 0
    offset_b = 0
    colors = [((77+offset_r)/255, (157+offset_g)/255, (210+offset_b)/255),
              (1, 1, 1),
              ((255+offset_r)/255, (177+offset_g)/255, (106+offset_b)/255)]
    cm = LinearSegmentedColormap.from_list('name', colors, N=n_bin)

    for _ in range(0, 1):
        contour = ax.contourf(xx, yy, probs, n_bin, cmap=cm, antialiased=False, vmin=0, vmax=1, alpha=0.3, linewidth=0,
                              linestyles='dashed', zorder=-1)
        ax.contour(contour, cmap=cm, linewidth=1, linestyles='dashed', zorder=-1, alpha=1)

    ax_c = fig.colorbar(contour)

    ax_c.set_alpha(1)
    ax_c.draw_all()

    ax_c.set_label("$P(y = 1)$")
    # ax_c.set_ticks([0, .25, 0.5, 0.75, 1])
    # ax_c.ax.set_yticklabels(['0', '0.15', '0.3', '0.45', '0.6', '0.75', '0.9', '1'])

    X_lda_0 = X_lda[y_lda == 0]
    X_lda_1 = X_lda[y_lda == 1]

    X_lda_0_t = X_test[y_test == 0]
    X_lda_1_t = X_test[y_test == 1]
    marker_size = 150
    ax.scatter(X_lda_0[:, 0], X_lda_0[:, 1], c=(39/255, 111/255, 158/255), s=marker_size, vmin=-.2, vmax=1.2,
               edgecolor=(49/255, 121/255, 168/255), linewidth=0, marker='s', alpha=0.7, label='Class0 (Healthy)'
               , zorder=1)

    ax.scatter(X_lda_1[:, 0], X_lda_1[:, 1], c=(251/255, 119/255, 0/255), s=marker_size, vmin=-.2, vmax=1.2,
               edgecolor=(255/255, 129/255, 10/255), linewidth=0, marker='^', alpha=0.7, label='Class1 (Unhealthy)'
               , zorder=1)

    ax.scatter(X_lda_0_t[:, 0], X_lda_0_t[:, 1], s=marker_size-10, vmin=-.2, vmax=1.2,
               edgecolor="black", facecolors='none', label='Test data', zorder=1)

    ax.scatter(X_lda_1_t[:, 0], X_lda_1_t[:, 1], s=marker_size-10, vmin=-.2, vmax=1.2,
               edgecolor="black", facecolors='none', zorder=1)

    ax.set(xlabel="$X_1$", ylabel="$X_2$")

    ax.contour(xx, yy, probs, levels=[.5], cmap="Reds", vmin=0, vmax=.6, linewidth=0.1)

    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    handles, labels = ax.get_legend_handles_labels()
    db_line = Line2D([0], [0], color=(183/255, 37/255, 42/255), label='Decision boundary')
    handles.append(db_line)

    plt.legend(loc=2, fancybox=True, framealpha=0.4, handles=handles)
    plt.title(title)
    ttl = ax.title
    ttl.set_position([.57, 0.97])
    # plt.tight_layout()

    # path = filename + '\\' + str(resolution) + '\\'
    # path_file = path + "%d_p.png" % days
    # pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    # plt.savefig(path_file, bbox_inches='tight')

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


def plot_2D_decision_boundaries_(X, y, X_test, title, clf, folder=None, options=None, i=0):
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
    gamma = clf.gamma
    coef0 = clf.coef0
    cost = clf.C
    tolerance = clf.tol
    probability_ = clf.probability

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
    if output_dim not in [1, 2, 3]:
        raise ValueError("available dimension for features reduction are 1, 2 and 3.")
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
    if output_dim != 1:
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
    try:
        if dim_reduc == 'LDA':
            X_train, X_test, y_train, y_test = reduce_lda(n, X_train, X_test, y_train, y_test)

        if dim_reduc == 'PCA':
            X_train, X_test, y_train, y_test = reduce_pca(n, X_train, X_test, y_train, y_test)

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


def compute_model(X, y, train_index, test_index, i, clf=None, dim=None, dim_reduc_name=None, clf_name='',
                  folder=None, options=None, resolution=None, enalble_1Dplot=True,
                  enalble_2Dplot=True, enalble_3Dplot=True, nfold=1):

    X_lda, y_lda, X_train, X_test, y_train, y_test = process_fold(dim, X, y, train_index, test_index,
                                                                  dim_reduc=dim_reduc_name)
    if X_lda is None:
        simplified_results = {"accuracy": -1,
                              "specificity": -1,
                              "recall": -1,
                              "precision": -1,
                              "proba_y_true": -1,
                              "proba_y_false": -1,
                              "f-score": -1}
        return -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, "empty", "empty", simplified_results, -1, -1

    print(clf_name, "null" if dim is None else dim, X_train.shape, "fitting...")
    clf.fit(X_train, y_train)
    print("Best estimator found by grid search:")
    # print(clf.best_estimator_)

    y_pred = clf.predict(X_test)
    # y_pred_val = clf.predict(X_val)
    y_probas = clf.predict_proba(X_test)
    p_y_true, p_y_false = get_proba(y_probas, y_pred)
    acc = accuracy_score(y_test, y_pred)
    # acc_val = accuracy_score(y_val, y_pred_val)
    print(classification_report(y_test, y_pred))

    precision_false, precision_true, recall_false, recall_true, fscore_false, fscore_true,\
    support_false, support_true = get_prec_recall_fscore_support(
        y_test, y_pred)

    if 'MLP' in clf_name and hasattr(clf, "hidden_layer_sizes"):
        clf_name = "%s%s" % (clf_name, str(clf.best_estimator_.hidden_layer_sizes).replace(' ', ''))

    if 'KNN' in clf_name and hasattr(clf, "n_neighbors"):
        clf_name = "%s%s" % (clf_name, str(clf.n_neighbors).replace(' ', ''))

    print((
                clf_name, '' if dim_reduc_name is None else dim_reduc_name, dim, nfold, i,
                acc * 100, precision_false * 100, precision_true * 100, recall_false * 100, recall_true * 100,
                p_y_false*100, p_y_true*100,
                np.count_nonzero(y_lda == 0), np.count_nonzero(y_lda == 1),
                np.count_nonzero(y_train == 0), np.count_nonzero(y_train == 1),
                np.count_nonzero(y_test == 0), np.count_nonzero(y_test == 1), resolution, ','.join(options)))

    title = '%s-%s %dD %dFCV\nfold_i=%d, acc=%.1f%%, p0=%d%%, p1=%d%%, r0=%d%%, r1=%d%%, p0=%d%%, p1=%d%%\ndataset: class0=%d;' \
            'class1=%d\ntraining: class0=%d; class1=%d\ntesting: class0=%d; class1=%d\nresolution=%s input=%s \n' % (
                clf_name, '' if dim_reduc_name is None else dim_reduc_name, dim, nfold, i,
                acc * 100, precision_false * 100, precision_true * 100, recall_false * 100, recall_true * 100,
                p_y_false*100, p_y_true*100,
                np.count_nonzero(y_lda == 0), np.count_nonzero(y_lda == 1),
                np.count_nonzero(y_train == 0), np.count_nonzero(y_train == 1),
                np.count_nonzero(y_test == 0), np.count_nonzero(y_test == 1), resolution, ','.join(options))

    file_path = 'empty'

    try:
        if dim == 1 and enalble_1Dplot:
            file_path = plot_2D_decision_boundaries(X_lda, y_lda, X_test, y_test, title, clf, folder=folder, options=options, i=i)

        if dim == 2 and enalble_2Dplot:
            file_path = plot_2D_decision_boundaries(X_lda, y_lda, X_test, y_test, title, clf, folder=folder, options=options, i=i)

        if dim == 3 and enalble_3Dplot and 'LREG' not in clf_name and 'MLP' not in clf_name and 'KNN' not in clf_name:
                file_path = plot_3D_decision_boundaries(X_lda, y_lda, X_test, y_test, title, clf, folder=folder,
                                                        options=options, i=i)
    except Exception as e:
        print(e)


    simplified_results = {"accuracy": acc, "specificity": recall_false,
                          "proba_y_true": p_y_true,
                          "proba_y_false": p_y_false,
                          "recall": recall_score(y_test, y_pred, average='weighted'),
                          "precision": precision_score(y_test, y_pred, average='weighted'),
                          "f-score": f1_score(y_test, y_pred, average='weighted')}

    return X_lda, y_lda, title, acc, precision_false, precision_true, recall_false, recall_true, fscore_false, fscore_true, support_false, support_true, \
           title.split('\n')[0], file_path, simplified_results, p_y_false, p_y_true


def dict_mean(dict_list, proba_y_false_2d, proba_y_true_2d):
    mean_dict = {}
    if len(dict_list) > 0:
        for key in dict_list[0].keys():
            mean_dict[key] = (sum(d[key] for d in dict_list) / len(dict_list)) * 100

    mean_dict['proba_y_false_list'] = proba_y_false_2d,
    mean_dict['proba_y_true_list'] = proba_y_true_2d
    return mean_dict


def get_proba(y_probas, y_pred):
    class_0 = []
    class_1 = []
    for i, item in enumerate(y_probas):
        if y_pred[i] == 0:
            class_0.append(item[0])
        if y_pred[i] == 1:
            class_1.append(item[1])

    class_0 = np.asarray(class_0)
    class_1 = np.asarray(class_1)

    proba_0 = np.mean(class_0) if class_0.size > 0 else 0
    proba_1 = np.mean(class_1) if class_1.size > 0 else 0

    if np.isnan(proba_0):
        proba_0 = 0
    if np.isnan(proba_1):
        proba_1 = 0

    return proba_0 , proba_1


def process(data_frame, fold=3, dim_reduc=None, clf_name=None, folder=None, options=None, resolution=None, y_col='label'):
    if clf_name not in ['SVM', 'MLP', 'LREG', 'KNN']:
        raise ValueError('classifier %s is not available! available clf_name are KNN, MPL, LREG, SVM' % clf_name)
    print("process...")
    try:
        X, y = process_data_frame(data_frame, y_col=y_col)
    except ValueError as e:
        print(e)
        print(data_frame, dim_reduc, clf_name, folder, options, resolution, y_col)
        return {"error": str(e)}
    # kf = StratifiedKFold(n_splits=fold, random_state=None, shuffle=True)
    # kf.get_n_splits(X)
    rkf = RepeatedKFold(n_splits=10, n_repeats=100, random_state=int((datetime.now().microsecond)/10))

    scores, scores_1d, scores_2d, scores_3d = [], [], [], []
    precision_false, precision_false_1d, precision_false_2d, precision_false_3d = [], [], [], []
    precision_true, precision_true_1d, precision_true_2d, precision_true_3d = [], [], [], []
    recall_false, recall_false_1d, recall_false_2d, recall_false_3d = [], [], [], []
    recall_true, recall_true_1d, recall_true_2d, recall_true_3d = [], [], [], []
    fscore_false, fscore_false_1d, fscore_false_2d, fscore_false_3d = [], [], [], []
    fscore_true, fscore_true_1d, fscore_true_2d, fscore_true_3d = [], [], [], []
    support_false, support_false_1d, support_false_2d, support_false_3d = [], [], [], []
    support_true, support_true_1d, support_true_2d, support_true_3d = [], [], [], []
    simplified_results_full, simplified_results_1d, simplified_results_2d, simplified_results_3d = [], [], [], []
    proba_y_false, proba_y_true , proba_y_false_2d, proba_y_true_2d = [], [], [], []
    clf_name_full, clf_name_1d, clf_name_2d, clf_name_3d = '', '', '', ''
    file_path_1d, file_path_2d, file_path_3d, file_path = '', '', '', ''
    clf = None
    if clf_name == 'SVM':
        param_grid = {'C': np.logspace(-6, -1, 10), 'gamma': np.logspace(-6, -1, 10)}
        # clf = GridSearchCV(SVC(kernel='linear', probability=True), param_grid, n_jobs=2)
        clf = SVC(kernel='linear', probability=True)

    if clf_name == 'LREG':
        param_grid = {'penalty': ['none', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        clf = GridSearchCV(LogisticRegression(random_state=int((datetime.now().microsecond)/10), solver='lbfgs', multi_class='multinomial', max_iter=100000), param_grid, n_jobs=2)

    if clf_name == 'KNN':
        param_grid = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        clf = GridSearchCV(KNeighborsClassifier(), param_grid)

    if clf_name == 'MLP':
        param_grid = {'hidden_layer_sizes': [(5, 2), (5, 3), (5, 4), (5, 5), (4, 2), (4, 3), (4, 4), (2, 2), (3, 3)],
                      'alpha': [1e-8, 1e-8, 1e-10, 1e-11, 1e-12]}
        clf = GridSearchCV(MLPClassifier(solver='sgd', random_state=1, max_iter=2000), param_grid)

    print("looking for best hyperparameters...")
    try:
        clf.fit(X, y)
    except ValueError as e:
        print(e)
        return {}
    # clf = clf.best_estimator_
    print(clf)

    fig_roc_2d, ax_roc_2d = plt.subplots()
    mean_fpr_2d = np.linspace(0, 1, 100)
    tprs_2d = []
    aucs_2d = []

    for i, (train_index, test_index) in enumerate(rkf.split(X)):
        if dim_reduc is None:
            X_lda, y_lda, title, acc, p_false, p_true, r_false, r_true, fs_false, fs_true, s_false, s_true, clf_name_full, file_path, sr, p_y_false, p_y_true = compute_model(
                X, y, train_index, test_index, i, clf=clf, clf_name=clf_name,
                folder=folder, options=options, resolution=resolution, nfold=fold)
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
            proba_y_false.append(p_y_false)
            proba_y_true.append(p_y_true)

        if dim_reduc is not None:
            # acc_1d, p_false_1d, p_true_1d, r_false_1d, r_true_1d, fs_false_1d, fs_true_1d, s_false_1d, s_true_1d,\
            # clf_name_1d, file_path_1d, sr_1d = compute_model(
            #     X, y, train_index, test_index, i, clf=clf, dim=1, dim_reduc_name=dim_reduc,
            #     clf_name=clf_name, folder=folder, options=options, resolution=resolution, nfold=fold)

            X_lda_2d, y_lda_2d, title_2d, acc_2d, p_false_2d, p_true_2d, r_false_2d, r_true_2d, fs_false_2d, fs_true_2d, s_false_2d, s_true_2d,\
            clf_name_2d, file_path_2d, sr_2d, pr_y_false_2d, pr_y_true_2d = compute_model(
                X, y, train_index, test_index, i, clf=clf, dim=2, dim_reduc_name=dim_reduc,
                clf_name=clf_name, folder=folder, options=options, resolution=resolution, nfold=fold)

            # acc_3d, p_false_3d, p_true_3d, r_false_3d, r_true_3d, fs_false_3d, fs_true_3d, s_false_3d, s_true_3d,\
            # clf_name_3d, file_path_3d, sr_3d = compute_model(
            #     X, y, train_index, test_index, i, clf=clf, dim=3, dim_reduc_name=dim_reduc,
            #     clf_name=clf_name, folder=folder, options=options, resolution=resolution, nfold=fold)
            # scores_1d.append(acc_1d)
            # precision_false_1d.append(p_false_1d)
            # precision_true_1d.append(p_true_1d)
            # recall_false_1d.append(r_false_1d)
            # recall_true_1d.append(r_true_1d)
            # fscore_false_1d.append(fs_false_1d)
            # fscore_true_1d.append(fs_true_1d)
            # support_false_1d.append(s_false_1d)
            # support_true_1d.append(s_true_1d)
            # simplified_results_1d.append(sr_1d)

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
            proba_y_false_2d.append(pr_y_false_2d)
            proba_y_true_2d.append(pr_y_true_2d)
            viz = plot_roc_curve(clf, X_lda_2d, y_lda_2d,
                                 name='',
                                 label='_Hidden',
                                 alpha=0, lw=1, ax=ax_roc_2d)
            interp_tpr = interp(mean_fpr_2d, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs_2d.append(interp_tpr)
            aucs_2d.append(viz.roc_auc)

            # scores_3d.append(acc_3d)
            # precision_false_3d.append(p_false_3d)
            # precision_true_3d.append(p_true_3d)
            # recall_false_3d.append(r_false_3d)
            # recall_true_3d.append(r_true_3d)
            # fscore_false_3d.append(fs_false_3d)
            # fscore_true_3d.append(fs_true_3d)
            # support_false_3d.append(s_false_3d)
            # support_true_3d.append(s_true_3d)
            # simplified_results_3d.append(sr_3d)

    plot_roc_range(ax_roc_2d, tprs_2d, mean_fpr_2d, aucs_2d, fig_roc_2d, title_2d, options, folder)
    fig_roc_2d.clear()
    print("svc %d fold cross validation 2d is %f, 3d is %s." % (
        fold, float(np.mean(scores_2d)), float(np.mean(scores_3d))))

    if dim_reduc is None:
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
                'support_false': np.mean(support_false),
                'proba_y_false': np.mean(proba_y_false_2d),
                'proba_y_true': np.mean(proba_y_true_2d),
                'proba_y_false_list': proba_y_false_2d,
                'proba_y_true_list': proba_y_true_2d
            },
            'simplified_results': dict_mean(simplified_results_full, proba_y_false_2d, proba_y_true_2d)
        }
    else:
        result = {
            'fold': fold,
            '1d_reduced' if len(scores_1d) > 0 else '1d_reduced_empty': {
                'db_file_path': file_path_1d,
                'clf_name': clf_name_1d,
                'scores': scores_1d,
                'accuracy': float(np.mean(scores_1d)),
                'precision_true': float(np.mean(precision_true_1d)),
                'precision_false': np.mean(precision_false_1d),
                'recall_true': float(np.mean(recall_true_1d)),
                'recall_false': np.mean(recall_false_1d),
                'fscore_true': float(np.mean(fscore_true_1d)),
                'fscore_false': float(np.mean(fscore_false_1d)),
                'support_true': np.mean(support_true_1d),
                'support_false': np.mean(support_false_1d),
                'proba_y_false': np.mean(proba_y_false_2d),
                'proba_y_true': np.mean(proba_y_true_2d),
                'proba_y_false_list': proba_y_false_2d,
                'proba_y_true_list': proba_y_true_2d

            },
            '2d_reduced' if len(scores_2d) > 0 else '2d_reduced_empty': {
                'db_file_path': file_path_2d,
                'clf_name': clf_name_2d,
                'scores': scores_2d,
                'accuracy': float(np.mean(scores_2d)),
                'precision_true': precision_true_2d,
                'precision_false': precision_false_2d,
                'recall_true': recall_true_2d,
                'recall_false': recall_false_2d,
                'fscore_true': fscore_true_2d,
                'fscore_false': fscore_false_2d,
                'support_true': np.mean(support_true_2d),
                'support_false': np.mean(support_false_2d),
                'proba_y_false': proba_y_false_2d,
                'proba_y_true': proba_y_true_2d
            },
            '3d_reduced' if len(scores_3d) > 0 else '3d_reduced_empty': {
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
                'support_false': np.mean(support_false_3d),
                'proba_y_false': np.mean(proba_y_false_2d),
                'proba_y_true': np.mean(proba_y_true_2d),
                'proba_y_false_list': proba_y_false_2d,
                'proba_y_true_list': proba_y_true_2d
            },
            'simplified_results': {
                'simplified_results_1d' if len(simplified_results_1d) > 0 else 'simplified_results_1d_empty': dict_mean(simplified_results_1d, proba_y_false_2d, proba_y_true_2d),
                'simplified_results_2d' if len(simplified_results_2d) > 0 else 'simplified_results_2d_empty': dict_mean(simplified_results_2d, proba_y_false_2d, proba_y_true_2d),
                'simplified_results_3d' if len(simplified_results_3d) > 0 else 'simplified_results_3d_empty': dict_mean(simplified_results_3d, proba_y_false_2d, proba_y_true_2d)
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
def find_type_for_mem_opt(df):
    data_col_n = df.iloc[[0]].size
    type_dict = {}
    for n, i in enumerate(range(0, data_col_n)):
        if n < (data_col_n - META_DATA_LENGTH):
            type_dict[str(i)] = np.float16
        else:
            type_dict[str(i)] = np.str
    del df
    type_dict[str(data_col_n-1)] = np.int
    type_dict[str(data_col_n-2)] = np.int
    type_dict[str(data_col_n-3)] = np.int
    type_dict[str(data_col_n-8)] = np.int
    type_dict[str(data_col_n - 9)] = np.int
    type_dict[str(data_col_n - 10)] = np.int
    type_dict[str(data_col_n - 11)] = np.int
    type_dict[str(data_col_n - 15)] = np.int
    return type_dict


def load_df_from_datasets(fname, label_col):
    df = pd.read_csv(fname, nrows=1, sep=",", header=None)
    print(df)
    type_dict = find_type_for_mem_opt(df)

    data_frame = pd.read_csv(fname, sep=",", header=None, dtype=type_dict, low_memory=False)
    print(data_frame)
    sample_count = df.shape[1]
    hearder = [str(n) for n in range(0, sample_count)]
    hearder[-16] = "label"
    hearder[-15] = "elem_in_row"
    hearder[-14] = "date1"
    hearder[-13] = "date2"
    hearder[-12] = "serial"
    hearder[-11] = "famacha_score"
    hearder[-10] = "previous_famacha_score"
    hearder[-9] = "previous_famacha_score2"
    hearder[-8] = "previous_famacha_score3"
    hearder[-8] = "previous_famacha_score4"

    hearder[-7] = "dtf1"
    hearder[-6] = "dtf2"
    hearder[-5] = "dtf3"
    hearder[-4] = "dtf4"
    hearder[-4] = "dtf5"

    hearder[-3] = "nd1"
    hearder[-2] = "nd2"
    hearder[-1] = "nd3"
    hearder[-1] = "nd4"

    data_frame.columns = hearder
    print(data_frame)
    data_frame_original = data_frame.copy()
    cols_to_keep = hearder[:-META_DATA_LENGTH]
    cols_to_keep.append(label_col)
    data_frame = data_frame[cols_to_keep]
    data_frame = shuffle(data_frame)
    return data_frame_original, data_frame, cols_to_keep


def process_classifiers(inputs, dir, resolution, dbt, thresh_nan, thresh_zeros, farm_id, sliding_w, label_col='label'):
    filename = init_result_file(dir, farm_id)
    filename_s = init_result_file(dir, farm_id, simplified_results=True)
    print("start classification...", inputs)
    start_time = time.time()
    for input in inputs:

        _, data_frame, _ = load_df_from_datasets(input["path"], label_col)
        print(data_frame)
        sample_count = data_frame.shape[1]
        try:
            class_true_count = data_frame[label_col].value_counts().to_dict()[True]
            class_false_count = data_frame[label_col].value_counts().to_dict()[False]
        except KeyError as e:
            print(e)
            continue
        print("class_true_count=%d and class_false_count=%d" % (class_true_count, class_false_count))
        print("current_file is", input)
        for result in [
            # process(data_frame, fold=5, dim_reduc='LDA', clf_name='SVM', folder=dir,
            #                   options=input["options"], resolution=resolution),
            # process(data_frame, fold=10, clf_name='SVM', folder=dir,
            #         options=input["options"], resolution=resolution)
            process(data_frame, fold=10, dim_reduc='LDA', clf_name='SVM', folder=dir, options=input["options"],
                    resolution=resolution)
            # process(data_frame, fold=5, dim_reduc='LDA', clf_name='KNN', folder=dir,
            #                   options=input["options"], resolution=resolution)
            # process(data_frame, fold=10, dim_reduc='LDA', clf_name='MLP', folder=dir,
            #         options=input["options"], resolution=resolution)
        ]:
            time_proc = get_elapsed_time_string(start_time, time.time())

            if result is None:
                continue

            if '1d_reduced' in result:
                r_1d = result['1d_reduced']
                append_result_file(filename, r_1d['accuracy'], r_1d['scores'], r_1d['precision_true'],
                                   r_1d['precision_false'],
                                   r_1d['recall_true'], r_1d['recall_false'], r_1d['fscore_true'],
                                   r_1d['fscore_false'], r_1d['support_true'], r_1d['support_false'],
                                   class_true_count,
                                   class_false_count, result['fold'], r_1d['proba_y_false'], r_1d['proba_y_true'],
                                   resolution, dbt, sliding_w, thresh_nan, thresh_zeros, time_proc, sample_count,
                                   data_frame.shape[0],
                                   input["path"], input["options"], r_1d['clf_name'], r_1d['db_file_path'])
            if '2d_reduced' in result:
                r_2d = result['2d_reduced']
                append_result_file(filename, r_2d['accuracy'], r_2d['scores'], r_2d['precision_true'],
                                   r_2d['precision_false'],
                                   r_2d['recall_true'], r_2d['recall_false'], r_2d['fscore_true'],
                                   r_2d['fscore_false'], r_2d['support_true'], r_2d['support_false'],
                                   class_true_count,
                                   class_false_count, result['fold'], r_2d['proba_y_false'], r_2d['proba_y_true'],
                                   resolution, dbt, sliding_w, thresh_nan, thresh_zeros, time_proc, sample_count,
                                   data_frame.shape[0],
                                   input["path"], input["options"], r_2d['clf_name'], r_2d['db_file_path'])
            if '3d_reduced' in result:
                r_3d = result['3d_reduced']
                append_result_file(filename, r_3d['accuracy'], r_3d['scores'], r_3d['precision_true'],
                                   r_3d['precision_false'],
                                   r_3d['recall_true'], r_3d['recall_false'], r_3d['fscore_true'],
                                   r_3d['fscore_false'], r_3d['support_true'], r_3d['support_false'],
                                   class_true_count,
                                   class_false_count, result['fold'], r_3d['proba_y_false'], r_3d['proba_y_true'],
                                   resolution, dbt, sliding_w, thresh_nan, thresh_zeros, time_proc, sample_count,
                                   data_frame.shape[0],
                                   input["path"], input["options"], r_3d['clf_name'], r_3d['db_file_path'])

            if 'not_reduced' in result:
                r = result['not_reduced']
                append_result_file(filename, r['accuracy'], r['scores'], r['precision_true'], r['precision_false'],
                                   r['recall_true'], r['recall_false'], r['fscore_true'],
                                   r['fscore_false'], r['support_true'], r['support_false'], class_true_count,
                                   class_false_count, result['fold'], r['proba_y_false'], r['proba_y_true'],
                                   resolution, dbt, sliding_w, thresh_nan, thresh_zeros, time_proc, sample_count,
                                   data_frame.shape[0],
                                   input["path"], input["options"], r['clf_name'], r['db_file_path'])

            if 'simplified_results' in result:
                if 'simplified_results_2d' in result['simplified_results']:
                    item = result['simplified_results']['simplified_results_2d']
                    append_simplified_result_file(filename_s, r_2d['clf_name'], item['accuracy'],
                                                  item['specificity'],
                                                  item['recall'], item['precision'], item['f-score'],
                                                  item['proba_y_false_list'], item['proba_y_true_list'],
                                                  dbt, sliding_w, resolution,
                                                  format_options(input["options"]))
                if 'simplified_results_1d' in result['simplified_results']:
                    item = result['simplified_results']['simplified_results_1d']
                    append_simplified_result_file(filename_s, r_1d['clf_name'], item['accuracy'],
                                                  item['specificity'],
                                                  item['recall'], item['precision'], item['f-score'],
                                                  item['proba_y_false_list'], item['proba_y_true_list'],
                                                  dbt, sliding_w, resolution,
                                                  format_options(input["options"]))
                if 'simplified_results_3d' in result['simplified_results']:
                    item = result['simplified_results']['simplified_results_3d']
                    append_simplified_result_file(filename_s, r_3d['clf_name'], item['accuracy'],
                                                  item['specificity'],
                                                  item['recall'], item['precision'], item['f-score'],
                                                  item['proba_y_false_list'], item['proba_y_true_list'],
                                                  dbt, sliding_w, resolution,
                                                  format_options(input["options"]))
                if 'simplified_results_full' in result['simplified_results']:
                    item = result['simplified_results']['simplified_results_full']
                    append_simplified_result_file(filename_s, r['clf_name'], item['accuracy'], item['specificity'],
                                                  item['recall'], item['precision'], item['f-score'],
                                                  item['proba_y_false_list'], item['proba_y_true_list'],
                                                  dbt, sliding_w, resolution,
                                                  format_options(input["options"]))


def format_options(options):
    return '+'.join(options).replace('humidity', 'h').replace('temperature', 't').replace('activity', 'a').replace(
        'indexes', 'i')


def merge_results(filename=None, filter=None, simplified_report=False):
    print("merging results...")
    purge_file(filename)
    directory_path = os.getcwd().replace('C', 'E')
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
    header = RESULT_FILE_HEADER_SIMPLIFIED.split(',') if simplified_report else RESULT_FILE_HEADER.split(',')
    for item in header:
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


def process_day(params):
    days_before_famacha_test, resolution, sliding_w, farm_id, src, sql_db = params[0], params[1], params[2], params[3], params[4], connect_to_sql_database()
    threshold_nan_coef = 5
    threshold_zeros_coef = 2
    nan_threshold, zeros_threshold = 0, 0
    create_cwt_graph_enabled = True
    create_activity_graph_enabled = True
    weather_data = None

    if resolution == "min":
        threshold_nan_coef = 1.5
        threshold_zeros_coef = 1.5
    # if resolution == "day":
    #     days_before_famacha_test_l = [3, 4, 5, 6]
    expected_sample_count = get_expected_sample_count(resolution, days_before_famacha_test)

    # generate_training_sets(data_famacha_flattened)
    try:
        with open(os.path.join(__location__, '%s_weather.json' % farm_id.split('_')[0])) as f:
            weather_data = json.load(f)
    except FileNotFoundError as e:
        print("error while reading weather data file", e)
        exit()

    # data_famacha_dict = generate_table_from_xlsx('Lange-Henry-Debbie-Skaap-Jun-2016a.xlsx')
    # with open('C:\\Users\\fo18103\\PycharmProjects\\prediction_of_helminths_infection\\db_processor\\src\\delmas_famacha_data.json', 'a') as outfile:
    #     json.dump(data_famacha_dict, outfile)

    with open(
            'C:\\Users\\fo18103\\PycharmProjects\\prediction_of_helminths_infection\\db_processor\\src\\%s_famacha_data.json' %
            farm_id.split('_')[0], 'r') as fp:
        data_famacha_dict = json.load(fp)
        print(data_famacha_dict.keys())
        if 'cedara' in farm_id:
            data_famacha_dict = format_cedara_famacha_data(data_famacha_dict, sql_db)

    data_famacha_list = [y for x in data_famacha_dict.values() for y in x]
    results = []
    # herd_data = []
    dir = "%s/%s_sld_%d_dbt%d_%s" % (os.getcwd().replace('C', 'E'), resolution, sliding_w,
                                     days_before_famacha_test, farm_id)
    class_input_dict_file_path = dir + '/class_input_dict.json'
    # if False:
    if os.path.exists(class_input_dict_file_path):
        print('training sets already created skip to processing.')
        with open(class_input_dict_file_path, "r") as read_file:
            class_input_dict = json.load(read_file)
            try:
                shutil.rmtree(dir + "/analysis")
                shutil.rmtree(dir + "/decision_boundaries_graphs")
                shutil.rmtree(dir + "/roc_curve")
            except (OSError, FileNotFoundError) as e:
                print(e)
    else:
        print('start training sets creation...')
        try:
            shutil.rmtree(dir)
        except (OSError, FileNotFoundError) as e:
            print(e)
            # exit(-1)

        for i, curr_data_famacha in enumerate(data_famacha_list):
            try:
                result = get_training_data(sql_db, curr_data_famacha, i, data_famacha_list.copy(), data_famacha_dict, weather_data, resolution,
                                           days_before_famacha_test, expected_sample_count,
                                           farm_sql_table_id=farm_id, sliding_windows=sliding_w)
            except KeyError as e:
                print(e)

            if result is None:
                continue

            is_valid, nan_threshold, zeros_threshold, h = is_activity_data_valid(result["activity"],
                                                                              threshold_nan_coef,
                                                                              threshold_zeros_coef)
            result['is_valid'] = True
            result['entropy'] = h
            if not is_valid:
                result['is_valid'] = False

            result["nan_threshold"] = nan_threshold
            result["zeros_threshold"] = zeros_threshold
            results.append(result)

        skipped_class_false, skipped_class_true = process_famacha_var(results)

        class_input_dict = []
        for idx in range(len(results)):
            result = results[idx]
            if not result['is_valid']:
                results[idx] = None
                continue
            if result['ignore']:
                results[idx] = None
                continue
            pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
            filename = create_filename(result)
            if create_activity_graph_enabled:
                create_activity_graph(result["activity"], dir, filename,
                                      title=create_graph_title(result, "time"),
                                      sub_sub_folder=str(result['famacha_score_increase']))

            # cwt, coef, freqs, indexes_cwt = compute_cwt(result["activity"])
            if 'sd' in src or 'sp' in src:
                print('sd')
                cwt, coef, freqs, indexes_cwt, scales, delta_t, wavelet_type = compute_cwt(result["activity"])
            else:
                print('hd')
                cwt, coef, freqs, indexes_cwt, scales, delta_t, wavelet_type = compute_hd_cwt(result["activity"])

            # cwt_weight = process_weight(result["activity"], coef)
            result["cwt"] = cwt
            result["coef_shape"] = coef.shape
            # result["cwt_weight"] = cwt_weight
            result["indexes_cwt"] = indexes_cwt

            # herd_data.append(result['herd'])
            if create_cwt_graph_enabled:
                create_hd_cwt_graph(coef, dir, filename, title=create_graph_title(result, "freq"),
                                    sub_sub_folder=str(result['famacha_score_increase']), freqs=freqs)

            class_input_dict = create_training_sets(result, dir)  # warning! always returns the same result
            if not os.path.exists(class_input_dict_file_path):
                with open(class_input_dict_file_path, 'w') as fout:
                    json.dump(class_input_dict, fout)
            # remove item from stack
            results[idx] = None
            gc.collect()

    # herd_file_path = dir + '/%s_herd_activity.json' % farm_id
    # herd_file_path = herd_file_path.replace('/', '\\')
    # if not os.path.exists(herd_file_path):
    #     with open(herd_file_path, 'w') as fout:
    #         json.dump({'herd_activity': herd_data}, fout)

    process_classifiers(class_input_dict, dir, resolution, days_before_famacha_test, nan_threshold,
                        zeros_threshold, farm_id, sliding_w)
    sql_db.cursor().close()
    sql_db.close()


def process_sliding_w(params):
    start_time = time.time()
    zipped = zip(['10min'], itertools.repeat(params[0]), itertools.repeat(params[1]), itertools.repeat(params[2]))
    for i, item in enumerate(zipped):
        print("%d/%d res=%s farm=%s progress..." % (i, len(item), item[0], item[2]))
        # days_before_famacha_test_l = range(1, 35)
        days_before_famacha_test_l = [2, 7, 14, 28]
        resolution, sliding_w, farm_id, src_folder = item[0], item[1], item[2], item[3]
        pool = Pool(processes=3)
        pool.map(process_day, zip(days_before_famacha_test_l, itertools.repeat(resolution),
                                  itertools.repeat(sliding_w), itertools.repeat(farm_id), itertools.repeat(src_folder))
                 )
        pool.close()
        pool.join()
    print(get_elapsed_time_string(start_time, time.time()))


if __name__ == '__main__':
    freeze_support()
    print('args=', sys.argv)
    print("pandas", pd.__version__)

    src_folders = ["sd\\"]
    for src_folder in src_folders:
        os.chdir(os.path.dirname(__file__))
        pathlib.Path(src_folder).mkdir(parents=True, exist_ok=True)
        os.chdir(src_folder)
        for farm_id in ["delmas_70101200027", "cedara_70091100056"]:
            pool = NonDaemonicPool(processes=1)
            pool.map(process_sliding_w, zip([0], itertools.repeat(farm_id), itertools.repeat(src_folder)))
            pool.close()
            pool.join()

            merge_results(filename="%s_results_simplified_report_%s.xlsx" % (farm_id, run_timestamp),
                          filter='%s_results_simplified.csv' % farm_id,
                          simplified_report=True)
            merge_results(filename="%s_results_report_%s.xlsx" % (farm_id, run_timestamp),
                          filter='%s_results.csv' % farm_id)
