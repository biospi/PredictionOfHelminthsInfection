import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymysql
import pathlib
import matplotlib.dates as mdates
import pycwt as wavelet
import plotly.express as px
import time
import pywt
import math
from datetime import datetime, timedelta
from math import pi, sin, log, exp

from numpy import random
from scipy.signal import chirp

import scipy
import scipy.signal as signal
from sys import exit

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 150)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0


def find_herd_activity(df):
    to_ignore = [40101310352, 40101310013, 40101310039, 40101310050, 40101310095, 40101310106, 40101310115,
                  40101310125, 40101310142, 40101310145, 40101310157, 40101310310, 40101310336, 40101310347]
    for ignore in to_ignore:
        df = df[df.serial != ignore]
    df = df.iloc[:, :-5]

    df = df.mean(axis=1, skipna=True)
    return df.tolist()



def get_data_by_animal(fname=''):
    print("loading dataset...")
    # print(fname)
    df = pd.read_csv(fname, sep=",", header=None)
    # print(data_frame)
    sample_count = df.shape[1]
    hearder = [str(n) for n in range(0, sample_count)]
    hearder[-5] = "label"
    hearder[-4] = "elem_in_row"
    hearder[-3] = "date1"
    hearder[-2] = "date2"
    hearder[-1] = "serial"
    df.columns = hearder
    herd = find_herd_activity(df)
    df['date1'] = df.date1.str.replace('\'', '')
    df['date2'] = df.date2.str.replace('\'', '')
    df['datetime1'] = pd.to_datetime(df['date1'], format="%d/%m/%Y")
    df['datetime2'] = pd.to_datetime(df['date2'], format="%d/%m/%Y")
    df['label'] = df['label'].map({True: 1, False: 0})
    df.sort_values(by='datetime1')
    df = df.loc[:, ['serial', 'label', 'datetime1', 'datetime2']]
    return dict(tuple(df.groupby('serial'))), herd


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


def get_activity_data_in_range(date1, date2, animal_id):
    rows_activity = execute_sql_query(
        "SELECT timestamp, timestamp_s, serial_number, first_sensor_value FROM %s_resolution_%s"
        " WHERE timestamp BETWEEN %s AND %s AND serial_number = %s" %
        ("delmas_70101200027", "10min", date2, date1,
         animal_id))
    rows_herd_activity = execute_sql_query(
        "SELECT timestamp, timestamp_s, serial_number, first_sensor_value FROM %s_resolution_%s"
        " WHERE timestamp BETWEEN %s AND %s AND serial_number = %s" %
        ("delmas_70101200027", "10min", date2, date1,
         '50000000000'))
    return rows_activity, rows_herd_activity


def attribute_color(df):
    f_value = df[4]
    if f_value == 0:
        return 'green'
    if f_value == 1:
        return 'red'
    if f_value == -1:
        return 'gray'


event_num = 0


def get_event(t):
    event_idx = ((t >= df_famacha.datetime2) & (t <= df_famacha.datetime1)).dot(np.arange(event_num))
    if ((t >= df_famacha.datetime2) & (t <= df_famacha.datetime1))[0] == False and event_idx == 0:
        # print(t, df_famacha, ((t >= df_famacha.datetime2) & (t <= df_famacha.datetime1)))
        return -1
    return df_famacha.label[event_idx]


def even_list(n):
    result = [1]
    for num in range(2, n * 2 + 1, 2):
        result.append(num)
    del result[-1]
    return np.asarray(result, dtype=np.int32)


def compute_cwt(activity):
    w = pywt.ContinuousWavelet('morl')

    scales = even_list(40)
    sampling_frequency = 1 / 60
    sampling_period = 1 / sampling_frequency
    activity_i = interpolate(activity)
    coef, freqs = pywt.cwt(np.asarray(activity_i), scales, w, sampling_period=sampling_period)
    cwt = [element for tupl in coef for element in tupl]
    indexes = np.asarray(list(range(coef.shape[1])))
    return cwt, coef, freqs, indexes, scales, 1, 'morlet'


def compute_cwt_hd(activity):
    print("compute_cwt...")
    # t, activity = dummy_sin()
    num_steps = len(activity)
    x = np.arange(num_steps)
    y = activity
    y = interpolate(y)

    delta_t = x[1] - x[0]
    scales = np.arange(1, num_steps + 1) / 1
    freqs = 1 / (wavelet.Morlet().flambda() * scales)
    wavelet_type = 'morlet'

    # y = [0 if x is np.nan else x for x in y] #todo fix
    coefs, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(y, delta_t, wavelet=wavelet_type, freqs=freqs)

    # print("*********************************************")
    # print(y)
    # print(coefs)
    # print("*******************************************")
    # iwave = wavelet.icwt(coefs, scales, delta_t, wavelet=wavelet_type)
    # plt.plot(iwave)
    # plt.show()
    # plt.plot(activity)
    # plt.show()
    #
    # plt.matshow((coefs.real))
    # plt.show()
    # exit()
    cwt = [element for tupl in coefs.real for element in tupl]
    indexes = np.asarray(list(range(len(coefs.real))))
    return cwt, coefs.real, freqs, indexes, scales, delta_t, wavelet_type


def interpolate(input_activity):
    try:
        i = np.array(input_activity, dtype=np.float)
        s = pd.Series(i)
        s = s.interpolate(method='linear', limit_direction='both')
        # s = s.interpolate(method='spline', limit_direction='both')
        return s.tolist()
    except ValueError as e:
        print(e)
        return input_activity


def normalized(v):
    return v / np.sqrt(np.sum(v ** 2))


def date_to_epoch_string(date):
    return str(date.timestamp()).split('.')[0]


def dummy_signal_random():
    y = [random.randint(0, 100) for x in range(100)]
    plt.plot(y)
    return y


def dummy_signal():
    T = 5
    n = 1000
    t = np.linspace(0, T, n, endpoint=False)
    f0 = 1
    f1 = 10
    y = chirp(t, f0, T, f1, method='logarithmic')
    y = [math.fabs(x) for x in y]
    plt.plot(t, y)
    plt.grid(alpha=0.25)
    plt.xlabel('t (seconds)')
    plt.show()
    return y


def find_max(data):
    max_index = float("-inf")
    max_value = float("-inf")

    for index, value in enumerate(data):
        if value > max_value:
            max_value = value
            max_index = index
        elif value == max_value:
            max_index = max(max_index, index)
    return max_value, max_index


def replace_n_max(data, n=10):
    for _ in range(n):
        data[find_max(data)[1]] = np.nan
    return data


def anscombe(value):
    return 2 * math.sqrt(value + (3 / 8))


def anscombe_log_list(activity):
    return [math.log(anscombe(x)) if x is not None else None for x in activity]


if __name__ == '__main__':
    parent_dir = "health_graphs_6"
    dict_herd =None
    pathlib.Path(parent_dir).mkdir(parents=True, exist_ok=True)
    dir = 'C:/Users/fo18103/PycharmProjects/prediction_of_helminths_infection/training_data_generator_and_ml_classifier/src/resolution_10min_days_6/'
    dfs, herd = get_data_by_animal(fname="%s/training_sets/activity_.data" % dir)

    for id, df_famacha in dfs.items():
        if id in [40101310352, 40101310013, 40101310039, 40101310050, 40101310095, 40101310106, 40101310115,
                  40101310125, 40101310142, 40101310145, 40101310157, 40101310310, 40101310336, 40101310347]:
            continue

        pathlib.Path("%s\\%s" % (parent_dir, str(id))).mkdir(exist_ok=True)
        df_famacha = df_famacha.reset_index(drop=True)
        event_num = len(df_famacha.label)

        d_start = df_famacha['datetime1'].max()
        cpt = 0
        while True:
            fig, axs = plt.subplots(3)
            fig.set_size_inches(38.40, 21.60)
            d_end = d_start - timedelta(days=6 * 2)

            if d_end < df_famacha["datetime2"][0]:
                break

            dict_activity, dict_herd = get_activity_data_in_range(date_to_epoch_string(d_start),
                                                       date_to_epoch_string(d_end),
                                                       str(id))
            print(dict_activity)
            dict_famacha = dict([(d1, {'date1': d1, 'date2': d2, 'label': l}) for d1, d2, l in
                                 zip(df_famacha.datetime1, df_famacha.datetime2, df_famacha.label)])
            for value_a in dict_activity:
                value_a['label'] = -1
                for _, value_f in dict_famacha.items():
                    date_to_check = datetime.strptime(value_a['timestamp_s'], '%Y-%m-%dT%H:%M')
                    date1 = value_f['date1'].to_pydatetime()
                    date2 = value_f['date2'].to_pydatetime()
                    if date2 <= date_to_check <= date1:
                        value_a['label'] = value_f['label']

            df_activity = pd.DataFrame(dict_activity)
            df_herd = pd.DataFrame(dict_herd)

            try:
                df_activity['day'], df_activity['time'] = df_activity['timestamp_s'].str.split('T', 1).str
            except KeyError as e:
                print(e)
                break

            df_activity['date_time'] = pd.to_datetime(df_activity.day, format='%Y-%m-%d')

            df_activity['timestamp_ss'] = pd.to_datetime(df_activity.timestamp_s)

            # df_activity["label"] = df_activity.date_time.transform(get_event)
            df_activity['color'] = df_activity.apply(attribute_color, axis=1)

            print(df_famacha)
            print(df_activity)
            print(set(df_activity['label'].tolist()))

            activity = df_activity['first_sensor_value'].to_numpy(np.float)
            activity[activity < 0] = np.nan

            herd = df_herd['first_sensor_value'].to_numpy(np.float)
            herd[herd < 0] = np.nan
            herd = anscombe_log_list(herd)

            # activity = np.clip(activity, 0, 1000)
            try:
                activity = replace_n_max(activity)
            except IndexError as e:
                print(e)
                break

            activity = activity.tolist()
            activity = np.divide(activity, herd)

            date_time = df_activity['timestamp_ss'].tolist()
            colors = df_activity['color'].tolist()

            # plt.plot(activity)
            # plt.show()
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
            plt.setp(plt.gca().xaxis.get_majorticklabels(), 'rotation', 90)
            axs[0].bar(date_time, activity, color=colors, align='edge', width=0.02)
            labels = ['healthy', 'unhealthy']
            colors_d = {'healthy': 'green', 'unhealthy': 'red'}
            handles = [plt.Rectangle((0, 0), 1, 1, color=colors_d[label]) for label in labels]
            axs[0].legend(handles, labels)
            axs[0].set_ylabel('Activity count')

            # axs[1].plot(date_time, activity)
            cwt, coefs, freqs, indexes, scales, delta_t, wavelet_type = compute_cwt(activity)

            c0 = np.reshape(normalized(coefs.real), coefs.shape)
            iwave0 = wavelet.icwt(c0, scales, delta_t, wavelet=wavelet_type)
            iwave0 = np.real(iwave0)

            axs[1].pcolor(date_time, freqs, coefs)
            axs[1].set_ylabel('Frequency')
            axs[1].set_yscale('log')

            axs[2].plot(date_time, iwave0)
            axs[2].set_ylabel('Amplitude')
            fig.tight_layout()
            # fig.show()
            fig.savefig("%s\\%s\\%s_%d.png" % (parent_dir, str(id), str(id), cpt), format='png', dpi=100)
            plt.clf()
            plt.cla()
            d_start = d_end
            cpt += 1
