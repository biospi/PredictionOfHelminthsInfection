# -*- coding: utf-8 -*-
import base64
import glob
import json
import operator
import os
import statistics
import sys
from datetime import datetime, timedelta
from itertools import groupby
from multiprocessing import Process, Queue
from time import mktime
from time import strptime

import dash
import dash_core_components as dcc
import dash_html_components as html
import flask
import numpy as np
from numpy import diff
import plotly
import plotly.graph_objs as go
import pymysql
import requests
import tables
from dash.dependencies import Input, Output
from dateutil.relativedelta import *
from ipython_genutils.py3compat import xrange
from scipy import signal
#import PyWavelets ?? for pywt
import pywt
import pywt.data
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy import interpolate
from scipy.stats import entropy
from math import log, e
from pylab import figure, show, legend, ylabel

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

global sql_db
db_name = "south_africa"


def get_date_range(layout_data, herd=False):
    #print('get_date_range', layout_data)

    x_axis = 'xaxis'
    if 'xaxis2.range[0]' in layout_data:
        x_axis = 'xaxis2'

    if 'xaxis3.range[0]' in layout_data:
        x_axis = 'xaxis3'

    x_min_epoch = None
    x_max_epoch = None
    try:
        xaxis_autorange = bool(layout_data["%s.autorange" % x_axis])
    except (KeyError, TypeError, ValueError):
        xaxis_autorange = False
    try:
        auto_range = layout_data["autosize"]
    except (KeyError, TypeError, ValueError):
        auto_range = False
    try:
        x_min = str(layout_data["%s.range[0]" % x_axis])
        if len(x_min.split(":")) == 2:
            x_min = x_min + ":00"
        if "." not in x_min:
            x_min = x_min + ".00"
        x_min_epoch = int(mktime(strptime(x_min, '%Y-%m-%d %H:%M:%S.%f')))
    except (KeyError, TypeError, ValueError):
        x_min = None
    try:
        x_max = str(layout_data["%s.range[1]" % x_axis])
        if len(x_max.split(":")) == 2:
            x_max = x_max + ":00"
        if "." not in x_max:
            x_max = x_max + ".00"
        x_max_epoch = int(mktime(strptime(x_max, '%Y-%m-%d %H:%M:%S.%f')))
    except (KeyError, TypeError, ValueError):
        x_max = None

    try:
        y_min = str(layout_data["yaxis.range[0]"])
        if len(y_min.split(":")) == 2:
            y_min = y_min + ":00"
        if "." not in y_min:
            y_min = y_min + ".00"
    except (KeyError, TypeError, ValueError):
        y_min = None
    try:
        y_max = str(layout_data["yaxis.range[1]"])
        if len(y_max.split(":")) == 2:
            y_max = y_max + ":00"
        if "." not in y_max:
            y_max = y_max + ".00"
    except (KeyError, TypeError, ValueError):
        y_max = None

    if x_max_epoch is None and x_min_epoch is None:
        return {'xaxis_autorange': True, 'auto_range': True}

    if herd:
        return {'x_min_epoch': x_min_epoch, 'x_max_epoch': x_max_epoch,
                'x_min': x_min, 'x_max': x_max,
                'y_min': y_min, 'y_max': y_max,
                'xaxis_autorange': xaxis_autorange, 'auto_range': auto_range}
    else:
        return {'x_min_epoch': x_min_epoch, 'x_max_epoch': x_max_epoch,
                'x_min': x_min, 'x_max': x_max,
                'xaxis_autorange': xaxis_autorange, 'auto_range': auto_range}


def get_elapsed_time_string(time_initial, time_next):
    dt1 = datetime.fromtimestamp(time_initial)
    dt2 = datetime.fromtimestamp(time_next)
    rd = relativedelta(dt2, dt1)
    return '%d years %d months %d days %d hours %d minutes %d seconds' % (
        rd.years, rd.months, rd.days, rd.hours, rd.minutes, rd.seconds)


def get_elapsed_time_array(time_initial, time_next):
    dt1 = datetime.fromtimestamp(time_initial)
    dt2 = datetime.fromtimestamp(time_next)
    rd = relativedelta(dt2, dt1)
    return [rd.years, rd.months, rd.days, rd.hours, rd.minutes, rd.seconds]


def get_elapsed_time_seconds(time_initial, time_next):
    dt1 = datetime.fromtimestamp(time_initial)
    dt2 = datetime.fromtimestamp(time_next)
    result = (dt2 - dt1).total_seconds()
    return result


def find_appropriate_resolution(duration):
    value = None
    if 0 < duration <= 3 * 3600.0:
        value = 5
    if 3 * 3600.0 < duration <= 4 * 3600.0:
        value = 5
    if 4 * 3600.0 < duration <= 259200.0:
        value = 3
    if 259200.0 < duration <= 604800.0:
        value = 3
    if 604800.0 < duration <= 5 * 604800.0:
        value = 2
    if 5 * 604800.0 < duration <= 10 * 604800.0:
        value = 1
    if duration > 10 * 604800.0:
        value = 1
    return value


def compare_dates(d1, d2):
    d1_ = datetime.strptime(d1, '%d/%m/%Y').strftime('%Y-%m-%d')
    d2_ = d2.split('T')[0]
    return d1_ == d2_


def chunks(l, n):
    n = max(1, n)
    return (l[i:i + n] for i in xrange(0, len(l), n))


def is_in_period(start, famacha_day, n):
    datetime_start = datetime.strptime(start, '%Y-%m-%d')
    datetime_famacha = datetime.strptime(famacha_day, '%d/%m/%Y')
    margin = timedelta(days=n)
    return datetime_start - margin <= datetime_famacha <= datetime_start + margin


def build_weather_trace(time, data_f):
    try:
        #print("weather data available for [%s]" % ','.join(data_f['weather'].keys()))
        weather_humidity = [None] * len(time)
        weather_temperature = [None] * len(time)
        date_weather = data_f['weather'].keys()
        for i, t in enumerate(time):
            day = t.split('T')[0]
            if day in date_weather:
                list = data_f['weather'][day]
                mean_humidity = 0
                mean_temperature = 0
                for item in list:
                    mean_humidity += int(item['humidity'])
                    mean_temperature += int(item['temp_c'])
                weather_humidity[i] = mean_humidity / len(list)
                weather_temperature[i] = mean_temperature / len(list)

        graph_h = go.Scatter(
            x=time,
            y=weather_humidity,
            name='humidity',
            opacity=0.1,
            yaxis='y2',
            fill='tozeroy',
            mode='none',
            fillcolor='rgba(76, 146, 195, .5)'
        )

        graph_t = go.Scatter(
            x=time,
            y=weather_temperature,
            name='temperature',
            opacity=0.1,
            yaxis='y3',
            fill='tozeroy',
            mode='none',
            fillcolor='rgba(255, 127, 39, 0.4)'
        )

        # data = [
        #     go.Scatter(
        #         x=time,
        #         y=weather_humidity,
        #         name='humidity',
        #         connectgaps=True
        #     ),
        #     go.Scatter(
        #         x=time,
        #         y=weather_temperature,
        #         name='temperature',
        #         yaxis='y2',
        #         connectgaps=True
        #     )]
        #
        # layout = go.Layout(
        #     title='Humidity and temperature over the collected data time period.',
        #     font=dict(size=20),
        #     yaxis=dict(
        #         title='Humidity (%)'
        #     ),
        #     yaxis2=dict(
        #         title='Temperature (°C)',
        #         titlefont=dict(
        #             color='rgb(148, 103, 189)'
        #         ),
        #         tickfont=dict(
        #             color='rgb(148, 103, 189)'
        #         ),
        #         overlaying='y',
        #         side='right'
        #     )
        # )
        # fig = go.Figure(data=data, layout=layout)
        # plotly.plotly.iplot(fig, filename='multiple-axes-double')

        return graph_h, graph_t

    except KeyError as e:
        print(e)


def build_famacha_trace(traces, data_f, resolution):
    traces_famacha = []
    try:
        for idx, t in enumerate(traces):
            time = t['x']
            famacha_s = [None] * len(time)
            serial = t['name']
            if serial not in data_f['famacha']:
                continue
            date_ = data_f['famacha'][serial].keys()
            date_famacha = [datetime.strptime(d, '%d/%m/%Y').strftime('%Y-%m-%d') for d in date_]
            for i, t in enumerate(time):
                day = t.split('T')[0]
                if resolution == 'resolution_week':
                    for day_in_famacha in date_famacha:
                        key = datetime.strptime(day_in_famacha, '%Y-%m-%d').strftime('%d/%m/%Y')
                        if is_in_period(day, key, 5):
                            famacha_s[i] = data_f['famacha'][serial][key]
                else:
                    if day in date_famacha:
                        key = datetime.strptime(day, '%Y-%m-%d').strftime('%d/%m/%Y')
                        famacha_s[i] = data_f['famacha'][serial][key]

            if idx == 0:
                f_s = famacha_s
            else:
                f_s = [-x if x is not None else None for x in famacha_s]

            traces_famacha.append(go.Bar(
                x=[str(x) for x in time],
                y=f_s,
                name='famacha score',
                # connectgaps=True,
                opacity=0.4,
                xaxis='x2',
                yaxis='y2'

            ))
        return traces_famacha
    except (KeyError, TypeError) as e:
        print(e)


def interpolate(input_activity, m='cubic'):
    try:
        i = np.array(input_activity, dtype=np.float)
        s = pd.Series(i)
        s = s.interpolate(method=m, limit_direction='both')
        return s.tolist()
    except ValueError as e:
        return input_activity


def compute_derivative(activity, resolution):
    dx = 1
    if resolution == 'resolution_day':
        dx = 1440
    if resolution == 'resolution_hour':
        dx = 60
    if resolution == 'resolution_5min':
        dx = 5
    y = interpolate(activity)
    dy = diff(y) / dx
    return dy


def build_derivative_graph(activity, resolution, time, serial='None', idx=0):
    layout = go.Layout(
        yaxis={'title': 'derivative'},
        barmode='overlay',
        showlegend=False,
        legend=dict(y=0.98), margin=go.layout.Margin(l=60, r=50, t=5, b=40))
    dy = compute_derivative(activity, resolution)
    trace = go.Bar(
        x=time[:-1],
        y=dy,
        opacity=1 if idx == 0 else 0.7,
        name=str(serial),
    )

    return trace, layout


def build_histogram_graph(activity, serial='None', idx=0, bin=5):
    layout = go.Layout(
        yaxis={'title': 'value count'},
        showlegend=False,
        legend=dict(y=0.98), margin=go.layout.Margin(l=60, r=50, t=5, b=40))
    hist, occur = compute_histogram(activity, bin)

    trace = go.Bar(
        x=hist,
        y=[x if idx == 0 else -x for x in occur],
        name=str(serial),
    )

    return trace, layout


def get_traces_max_amplitude(traces):
    extrema = []
    for trace in traces:
        if type(trace['y']) is not list:
            continue
        try:
            extrema.append(math.fabs(float(max(filter(None, trace['y'])))))
        except ValueError as e:
            print(e)
            extrema.append(10)
        try:
            extrema.append(math.fabs(float(min(filter(None, trace['y'])))))
        except ValueError as e:
            print(e)
            extrema.append(-10)
    return max(extrema)


def build_activity_graph(data, data_f, dragmode):
    layout = None
    figure = None
    d = data[0]
    # x_max = d["x_max"]
    signal_size = d["signal_size"]
    min_activity_value = d["min_activity_value"]
    max_activity_value = d["max_activity_value"]
    start_date = d["start_date"]
    end_date = d["end_date"]
    time_range = d["time_range"]
    activity = d["activity"]
    time = d["time"]
    relayout_data = d["relayout_data"]
    traces = d["traces"]
    range_d = d["range_d"]
    resolution = d["resolution"]

    fig_famacha = build_famacha_trace(traces, data_f, resolution)
    if fig_famacha is not None:
        traces.extend(fig_famacha)

    # if x_max is not None:
    # x_axis_data = {'title': 'Time'}
    x_axis_data = {'showline': False, 'zeroline': False}
    x_max_epoch, x_min_epoch, x_max, x_min, xaxis_autorange, auto_range = parse_date_dange(range_d)
    if x_min is not None:
        x_axis_data['autorange'] = xaxis_autorange
        x_axis_data['range'] = [x_min, x_max]

    enable_dragmode = None
    if dragmode is not None and "dragmode" in dragmode:
        enable_dragmode = "pan"

    a = get_traces_max_amplitude(traces)

    layout = go.Layout(xaxis=x_axis_data,
                       yaxis=dict(title='Activity level/Accelerometer count', mirror=True, zeroline=False, range=[-a if len(traces) > 2 else 0, a]),
                       xaxis2=dict(showline=False, zeroline=False, showgrid=False),
                       yaxis2=dict(
                           nticks=3,
                           overlaying='y1',
                           range=[-2 if len(traces) > 2 else 0, 2],
                           # showgrid=False,
                           # zeroline=False,
                           side='right'
                       ),
                       dragmode=enable_dragmode,
                       autosize=range_d['auto_range'],
                       legend=dict(y=0.98),
                       margin=go.layout.Margin(l=60, r=50, t=5, b=40)
                       )
    figure = [{'thread_activity': True}, {'signal_size': signal_size}, {'min_activity_value': min_activity_value},
              {'max_activity_value': max_activity_value}, {'start_date': start_date}, {'resolution': resolution},
              {'end_date': end_date}, {'time_range': time_range},
              {'activity': activity}, {'time': time}, {'relayout_data': relayout_data}, {
                  'data': traces,
                  'layout': layout
              }]

    return figure, layout


def pad(l, size, padding):
    return l + [padding] * abs((len(l) - size))


def get_resolution_string(value):
    result = 'resolution_month'
    if value == 0:
        result = 'resolution_day'
    if value == 1:
        result = 'resolution_hour'
    if value == 2:
        result = 'resolution_10min'
    if value == 3:
        result = 'resolution_5min'
    if value == 4:
        result = 'resolution_min'
    if value == 5:
        result = 'resolution_min'
    return result


def anscombe(value):
    try:
        return math.log(2 * math.sqrt(value + (3 / 8)))
    except TypeError as e:
        print(e)


def myround(x, base=5):
    return base * round(x/base)


def compute_histogram(input_array, bin):
    filtered = []
    for item in input_array:
        if item is None:
            # filtered.append(-19)
            pass
        else:
            filtered.append(myround(int(item), base=bin))

    result = {}
    for item in filtered:
        occur = filtered.count(item)
        result[item] = occur
    return list(result.keys()), list(result.values())


# def normalize_activity_array_anscomb(activity):
#     result = []
#     for i in range(0, len(activity)):
#         try:
#             result.append(ascombe(activity[i]))
#         except (ValueError, TypeError) as e:
#             print('error while normalize_activity_array_ascomb', e)
#             result.append(None)
#     return result


# def normalize_activity_matrix_hmeandiff(activity_mean_l, activity_list):
#     result = []
#     for activity in activity_list:
#         a = normalize_histogram_mean_diff(activity_mean_l, activity)
#         result.append(a)
#     return result


# def normalize_activity_matrix_ascombe(activity_list):
#     result = []
#     for activity in activity_list:
#         r = []
#         for i, item in enumerate(activity):
#             n = ascombe(item)
#             r.append(n)
#         result.append(r)
#     return result


def parse_date_dange(range):
    x_max_epoch = None
    if 'x_max_epoch' in range:
        x_max_epoch = range['x_max_epoch']
    x_min_epoch = None
    if 'x_min_epoch' in range:
        x_min_epoch = range['x_min_epoch']
    x_max = None
    if 'x_max' in range:
        x_max = range['x_max']
    x_min = None
    if 'x_min' in range:
        x_min = range['x_min']
    xaxis_autorange = None
    if 'xaxis_autorange' in range:
        xaxis_autorange = range['xaxis_autorange']
    auto_range = None
    if 'auto_range' in range:
        auto_range = range['auto_range']

    return x_max_epoch, x_min_epoch, x_max, x_min, xaxis_autorange, auto_range


def entropy(labels, base=None):
    vc = pd.Series(labels).value_counts(normalize=True, sort=False)
    base = e if base is None else base
    return -(vc * np.log(vc)/np.log(base)).sum()


def process_sensor_value(value):
    return value
    if value is not None:
        return value if value < 10000 else 10000
    return value


def thread_activity_herd(q_4, intermediate_value, filter_famacha, relayout_data, selected_serial_number, normalize, activity_mean):
    data = None
    range_d = get_date_range(json.loads(relayout_data))
    x_max_epoch, x_min_epoch, x_max, x_min, xaxis_autorange, auto_range = parse_date_dange(range_d)
    resolution_string = 'resolution_day'
    value = None
    if 'x_max' in range_d and range_d['x_max'] is not None:
        value = find_appropriate_resolution(get_elapsed_time_seconds(x_min_epoch, x_max_epoch))
        resolution_string = get_resolution_string(value)

    if intermediate_value is not None:
        raw = json.loads(intermediate_value)

        file_path = raw["file_path"]
        farm_id = raw["farm_id"]

        if 'sql' == 'h5':
            print("opening file in thread test")
            h5 = tables.open_file(file_path, "r")
            data = [(datetime.utcfromtimestamp(x['timestamp']).strftime('%Y-%m-%dT%H:%M'),
                     x['first_sensor_value'])
                    for x in h5.root.resolution_month.data if x_min_epoch < x['timestamp'] < x_max_epoch]

        if 'sql' == 'sql':
            if x_max_epoch is not None:
                rows = execute_sql_query(
                    "SELECT timestamp_s, first_sensor_value, serial_number FROM %s_%s WHERE timestamp BETWEEN %s AND %s" %
                    (farm_id, resolution_string, x_min_epoch, x_max_epoch))
            else:
                rows = execute_sql_query(
                    "SELECT timestamp_s, first_sensor_value, serial_number FROM %s_%s" %
                    (farm_id, resolution_string))


            famacha = [40101310143, 40101310125, 40101310145, 40101310299, 40101310353, 40101310345, 40101310013,
                       40101310352, 40101310342, 40101310134, 40101310157, 40101310036, 40101310039, 40101310249,
                       40101310106, 40101310115, 40101310316, 40101310142, 40101310107, 40101310119, 40101310146,
                       40101310386, 40101310336, 40101310095, 40101310310, 40101310314, 40101310350, 40101310069,
                       40101310121, 40101310098, 40101310347, 40101310109, 40101310050]


            if 'cubic' in filter_famacha:
                data = [(x['timestamp_s'], process_sensor_value(x['first_sensor_value']), x['serial_number']) for x in rows if x['serial_number'] in famacha]
            else:
                data = [(x['timestamp_s'], process_sensor_value(x['first_sensor_value']), x['serial_number']) for x in rows]


    activity_list = []

    data_list = []

    if data is not None:
        data_list = [list(v) for l, v in groupby(sorted(data, key=lambda x: x[2]), lambda x: x[2])]

    records = []
    time = None

    max_size = 0
    for item in data_list:
        a = [(x[1]) for x in item]
        if max_size < len(a):
            max_size = len(a)

    for item in data_list:
        serial = item[0][2]
        a = [(x[1]) for x in item]
        # entropy_v = entropy(a)
        # if entropy_v < 3:
        #     continue
        a = pad(a, max_size, None)
        #print(len(a))
        time = [(x[0]) for x in item]

        a, time = bin_data_to_screen_size(a, time, 1000)

        records.append((a, serial))

    ids = []
    connect_gap = False

    traces_histogram = []
    layout_histogram = {}
    activity_mean_l = None
    if activity_mean is not None:
        activity_mean_l = json.loads(activity_mean)

    for i in records:
        a = i[0]
        ids.append(i[1])

        # if 'Ascomb' in normalize:
        #     a = normalize_activity_array_ascomb(a)

        # if 'HMeanDiff' in normalize:
        #     if i[1] != 50000000000 and i[1] != 60000000000:
        #         a = normalize_histogram_mean_diff(activity_mean_l, a)
        activity_list.append(a)
        t, layout_histogram = build_histogram_graph(a, i[1])
        traces_histogram.append(t)

        # if 'cubic' in cubic_interpolation:
        #     connect_gap = True
        #     # activity_list.append(interpolate(a))
        # else:
        #     connect_gap = False
        #     # activity_list.append(a)

    _d = []
    if len(activity_list) > 0:
        traces = []
        signal_size = len(activity_list[0])
        max_activity_value = 0
        min_activity_value = 0
        s_d = time[0]
        e_d = time[len(time) - 1]
        d1 = (datetime.strptime(s_d, '%Y-%m-%dT%H:%M') - datetime(1970, 1, 1)).total_seconds()
        d2 = (datetime.strptime(e_d, '%Y-%m-%dT%H:%M') - datetime(1970, 1, 1)).total_seconds()
        start_date = datetime.fromtimestamp(d1).strftime('%d/%m/%Y %H:%M:%S')
        end_date = datetime.fromtimestamp(d2).strftime('%d/%m/%Y %H:%M:%S')
        time_range = get_elapsed_time_string(d1, d2)
        if resolution_string == 'resolution_day':
            time = [t.split('T')[0] for t in time]

        serials = []
        for id in ids:
            string_id = ".." + str(id)[5:]
            if selected_serial_number == id:
                string_id = "<b>%s<b>" % (".." + str(id)[5:])
            serials.append(string_id)

        # print(normalize)
        # if 'HMeanDiff' in normalize:
        #     activity_list = normalize_activity_matrix_hmeandiff(activity_mean_l, activity_list)
        #
        # if 'Ascomb' in normalize:
        #     activity_list = normalize_activity_matrix_ascombe(activity_list)

        trace = go.Heatmap(z=activity_list,
                           x=time,
                           y=serials,
                           connectgaps=connect_gap,
                           colorscale='Viridis')
        traces.append(trace)

        _d.append({"activity": activity_list[0],
                   "time": time,
                   "range_d": range_d,
                   "start_date": start_date,
                   "end_date": end_date,
                   "signal_size": signal_size,
                   "min_activity_value": min_activity_value,
                   "max_activity_value": max_activity_value,
                   "time_range": time_range,
                   "traces": traces,
                   "traces_histogram": traces_histogram,
                   "layout_histogram": layout_histogram,
                   "x_max": x_max,
                   "x_min": x_min,
                   "relayout_data": relayout_data,
                   'resolution': resolution_string})
    q_4.put(_d)


def normalize_histogram_mean_diff(activity_mean, activity, flag=False, serial=None):
    # if flag:
    #     plt.plot(activity)
    #     plt.ylabel(str(serial))
    #     plt.show()
    scale = [0 for _ in range(0, len(activity))]
    idx = []
    for n, a in enumerate(activity):
        if a is None or a <= 0:
            continue
        if activity_mean[n] is None:
            continue
        r = (int(activity_mean[n]) - int(a))
        scale[n] = r
        idx.append(n)
    median = math.fabs(statistics.median(sorted(set(scale))))
    #print(scale)
    for i in idx:
        activity[i] = activity[i] * median
    return activity


def thread_activity(q_1, selected_serial_number, intermediate_value, normalize, cubic_interpolation, relayout_data, activity_mean):
    input_ag = []
    _d = []
    traces = []
    if isinstance(selected_serial_number, list):
        input_ag.extend(selected_serial_number)
    else:
        input_ag.append(selected_serial_number)
    if not selected_serial_number:
        print("selected_serial_number empty")
    else:
        # print("1 the selected serial number are: %s" % ', '.join(str(e) for e in input_ag))
        # print("1 value is %d" % value)
        activity_list = []
        activity_mean_l = None
        if activity_mean is not None:
            activity_mean_l = json.loads(activity_mean)
        for idx, i in enumerate(input_ag):
            data = None
            range_d = get_date_range(json.loads(relayout_data))
            x_max_epoch, x_min_epoch, x_max, x_min, xaxis_autorange, auto_range = parse_date_dange(range_d)

            resolution_string = 'resolution_day'
            if x_max is not None:
                value = find_appropriate_resolution(get_elapsed_time_seconds(x_min_epoch, x_max_epoch))
                resolution_string = get_resolution_string(value)
            raw = json.loads(intermediate_value)
            file_path = raw["file_path"]
            farm_id = raw["farm_id"]

            if 'sql' == 'h5':
                print("opening file in thread test")
                h5 = tables.open_file(file_path, "r")

            if 'sql' == 'h5':
                data = [(datetime.utcfromtimestamp(x['timestamp']).strftime('%Y-%m-%dT%H:%M'),
                         x['first_sensor_value'])
                        for x in h5.root.resolution_month.data if
                        x['serial_number'] == i and x_min_epoch < x['timestamp'] < x_max_epoch]
            if 'sql' == 'sql':
                if x_max_epoch is not None:
                    rows = execute_sql_query(
                        "SELECT timestamp, first_sensor_value FROM %s_%s WHERE serial_number=%s AND timestamp BETWEEN %s AND %s" %
                        (farm_id, resolution_string, i, x_min_epoch, x_max_epoch))
                else:
                    rows = execute_sql_query(
                        "SELECT timestamp, first_sensor_value FROM %s_%s WHERE serial_number=%s" %
                        (farm_id, resolution_string, i))

                data = [
                    (datetime.utcfromtimestamp(x['timestamp']).strftime('%Y-%m-%dT%H:%M'), x['first_sensor_value'])
                    for x in rows]

            activity = [(x[1]) for x in data]

            # print(activity_s)
            # activity = activity_s

            if 'Anscombe' in normalize:
                activity = [anscombe(x) if x is not None else None for x in activity]
                activity_mean_l = [anscombe(x) if x is not None else None for x in activity_mean_l]

            if 'HMeanDiff' in normalize:
                activity = normalize_histogram_mean_diff(activity_mean_l, activity, True, i)
            # activity, _ = compute_histogram(interpolate(activity))



            time = [(x[0]) for x in data]

            # print(activity)
            # print(time)

            activity, time = bin_data_to_screen_size(activity, time, 1000)

            # activity_without_herd = [a_i - b_i if a_i is not None and b_i is not None else None for a_i, b_i in zip(activity, activity_mean_l)]

            if len(activity) > 0:

                signal_size = len(activity)
                max_activity_value = 0
                min_activity_value = 0
                try:
                    max_activity_value = max(x for x in activity if x is not None)
                    min_activity_value = min(x for x in activity if x is not None)
                except ValueError as e:
                    print(e)

                s_d = time[0]
                e_d = time[len(time) - 1]
                d1 = (datetime.strptime(s_d, '%Y-%m-%dT%H:%M') - datetime(1970, 1, 1)).total_seconds()
                d2 = (datetime.strptime(e_d, '%Y-%m-%dT%H:%M') - datetime(1970, 1, 1)).total_seconds()
                start_date = datetime.fromtimestamp(d1).strftime('%d/%m/%Y %H:%M:%S')
                end_date = datetime.fromtimestamp(d2).strftime('%d/%m/%Y %H:%M:%S')
                time_range = get_elapsed_time_string(d1, d2)

                if resolution_string == 'resolution_day':
                    time = [t.split('T')[0] for t in time]

                # if 'cubic' in cubic_interpolation:
                #     activity = interpolate(activity)

                if idx == 1:
                    # fig = go.Bar(
                    #     xaxis='x2',
                    #     x=time,
                    #     y=[-x if x is not None else None for x in activity_without_herd],
                    #     name=str(i),
                    #     opacity=0.8,
                    # )
                    # traces.append(fig)

                    fig = go.Bar(
                        xaxis='x2',
                        x=time,
                        y=[-x if x is not None else None for x in activity],
                        name=str(i),
                        opacity=0.8,
                    )
                    traces.append(fig)
                    # if resolution_string != 'resolution_min':
                    #     fig_m = go.Scatter(
                    #         xaxis='x2',
                    #         x=time,
                    #         y=[-x if x is not None else None for x in activity_mean_l],
                    #         name='mean',
                    #         opacity=0.7,
                    #         mode='lines'
                    #     )
                    #     traces.append(fig_m)

                if idx == 0:

                    # fig = go.Bar(
                    #     xaxis='x2',
                    #     x=time,
                    #     y=activity_without_herd,
                    #     name=str(i),
                    #     opacity=0.8,
                    # )
                    # traces.append(fig)

                    fig = go.Bar(
                        xaxis='x2',
                        x=time,
                        y=activity,
                        name=str(i),
                        opacity=0.8,
                    )
                    traces.append(fig)
                    # if resolution_string != 'resolution_min':
                    #     fig_m = go.Scatter(
                    #         xaxis='x2',
                    #         x=time,
                    #         y=activity_mean_l,
                    #         name='mean',
                    #         opacity=0.7,
                    #         mode='lines'
                    #     )
                    #     traces.append(fig_m)

                _d.append({"activity": activity,
                           "time": time,
                           "range_d": range_d,
                           "start_date": start_date,
                           "end_date": end_date,
                           "signal_size": signal_size,
                           "min_activity_value": min_activity_value,
                           "max_activity_value": max_activity_value,
                           "time_range": time_range,
                           "traces": traces,
                           "x_max": x_max,
                           "relayout_data": relayout_data,
                           'resolution': resolution_string})
    q_1.put(_d)


def thread_signal(q_2, selected_serial_number, intermediate_value, relayout_data, temperature, humidity, signal_strength):
    if intermediate_value is None:
        selected_serial_number = []
    input_ss = []
    x_max = None
    if isinstance(selected_serial_number, list):
        input_ss.extend(selected_serial_number)
    else:
        input_ss.append(selected_serial_number)
    traces = []
    if not selected_serial_number:
        pass
    else:

        for idx, i in enumerate(input_ss):

            data = None
            range_d = get_date_range(json.loads(relayout_data))
            x_max_epoch, x_min_epoch, x_max, x_min, xaxis_autorange, auto_range = parse_date_dange(range_d)
            resolution_string = 'resolution_day'
            if x_max is not None:
                value = find_appropriate_resolution(get_elapsed_time_seconds(x_min_epoch, x_max_epoch))
                resolution_string = get_resolution_string(value)

            raw = json.loads(intermediate_value)

            file_path = raw["file_path"]
            farm_id = raw["farm_id"]
            if 'sql' == 'h5':
                print("opening file in thread signal")
                h5 = tables.open_file(file_path, "r")

            if 'sql' == 'h5':
                data = [(datetime.utcfromtimestamp(x['timestamp']).strftime('%Y-%m-%dT%H:%M'),
                         x['signal_strength_max'], x['signal_strength_min'])
                        for x in h5.root.resolution_month.data if
                        x['serial_number'] == i and x_min_epoch < x['timestamp'] < x_max_epoch]
            if 'sql' == 'sql':
                if resolution_string == 'resolution_min':
                    rows = execute_sql_query(
                        "SELECT timestamp, signal_strength FROM %s_%s WHERE serial_number=%s AND timestamp BETWEEN %s AND %s" %
                        (farm_id, resolution_string, i, x_min_epoch, x_max_epoch))
                    data = [
                        (datetime.utcfromtimestamp(x['timestamp']).strftime('%Y-%m-%dT%H:%M'), x['signal_strength'])
                        for x in rows]
                else:
                    if x_max_epoch is not None:
                        rows = execute_sql_query(
                            "SELECT timestamp, signal_strength_max, signal_strength_min FROM %s_%s WHERE serial_number=%s AND timestamp BETWEEN %s AND %s" %
                            (farm_id, resolution_string, i, x_min_epoch, x_max_epoch))
                    else:
                        rows = execute_sql_query(
                            "SELECT timestamp, signal_strength_max, signal_strength_min FROM %s_%s WHERE serial_number=%s" %
                            (farm_id, resolution_string, i))

                    data = [
                        (datetime.utcfromtimestamp(x['timestamp']).strftime('%Y-%m-%dT%H:%M'),
                         x['signal_strength_max'], x['signal_strength_min'])
                        for x in rows]

            time = [(x[0]) for x in data]
            if idx == 0:
                fig_humidity, fig_temperature = build_weather_trace(time, raw)
                print(humidity, temperature)
                if 'enabled' in humidity:
                    traces.append(fig_humidity)
                if 'enabled' in temperature:
                    traces.append(fig_temperature)

            if 'enabled' in signal_strength:
                if resolution_string is not 'resolution_min':
                    signal_strength_min = [(x[2]) for x in data]
                    signal_strength_max = [(x[1]) for x in data]
                    if signal_strength_min is not None:
                        traces.append(go.Scatter(
                            x=time,
                            y=[math.fabs(x) if x is not None else None for x in signal_strength_min],
                            mode='lines+markers',
                            marker=dict(size=3),
                            opacity=1,
                            name=("signal strength min %d" % i) if (len(input_ss) > 1) else "signal strength min"

                        ))
                    if signal_strength_max is not None:
                        traces.append(go.Scatter(
                            x=time,
                            y=[math.fabs(x) if x is not None else None for x in signal_strength_max],
                            mode='lines+markers',
                            marker=dict(size=3),
                            opacity=1,
                            name=("signal strength max %d" % i) if (len(input_ss) > 1) else "signal strength min"
                        ))
                else:
                    signal_strength_ = [(x[1]) for x in data]
                    if signal_strength_ is not None:
                        traces.append(go.Scatter(
                            x=time,
                            y=[math.fabs(x) if x is not None else None for x in signal_strength_],
                            mode='lines+markers',
                            marker=dict(size=3),
                            opacity=1,
                            name=("signal strength%d" % i) if (len(input_ss) > 1) else "signal strength"
                        ))
            else:
                traces.append([])

    if x_max is not None:
        q_2.put({
            'data': traces,
            'layout': go.Layout(xaxis={'autorange': xaxis_autorange,
                                       'range': [x_min, x_max]},
                                yaxis=dict(title='RSSI(received signal strength in)'),
                                legend=dict(
                                    traceorder='reversed'
                                ),
                                yaxis2=dict(
                                    anchor='x2',
                                    overlaying='y',
                                    side='right',
                                    title='Humidity(%)/Temp(°c)',
                                    ticks='',
                                    showticklabels=False
                                ),
                                yaxis3=dict(
                                    anchor='x',
                                    overlaying='y',
                                    side='right',
                                    ticks='',
                                    showticklabels=False
                                ),
                                autosize=auto_range,
                                showlegend=False,
                                # legend=dict(y=1, x=0),
                                margin=go.layout.Margin(l=60, r=50, t=5, b=40)
                                )
            # 'resolution': "resolution_day"
        })
    else:
        q_2.put({
            'data': traces,
            'layout': go.Layout(xaxis={'autorange': True}, yaxis={'title': 'RSSI(received signal '
                                                                           'strength in)',
                                                                  'autorange': True},
                                yaxis2=dict(
                                    anchor='x',
                                    overlaying='y',
                                    side='right',
                                    title='Humidity(%)/Temp(°c)',
                                    ticks='',
                                    showticklabels=False
                                ),
                                yaxis3=dict(
                                    anchor='x',
                                    overlaying='y',
                                    side='right',
                                    ticks='',
                                    showticklabels=False
                                ),
                                autosize=True,
                                showlegend=False,
                                # legend=dict(y=1, x=0),
                                margin=go.layout.Margin(l=60, r=50, t=5, b=40)
                                )
            # 'resolution': "resolution_day"
        })


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    return np.isnan(y), lambda z: z.nonzero()[0]


def interpolate_list(input):
    y = np.array(input, dtype=np.float64)
    nans, x = nan_helper(y)
    y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    return y.tolist()


def even_list(n):
    result = []
    for num in range(2, n * 2 + 1, 2):
        result.append(num)
    return np.asarray(result, dtype=np.int32)


def thread_spectrogram(q_3, window_size, radio, wavelet, data):
    traces = []
    traces_derivative = []
    activity_i = []
    time = []
    x_max = None
    range_d = None
    for idx, f in enumerate(data):
        activity = f["activity"]
        time = f["time"]
        relayout = f["relayout_data"]
        resolution = f["resolution"]

        j = json.loads(relayout)
        range_d = get_date_range(j)
        # print(activity, time, window_size, radio, relayout)
        x_max_epoch, x_min_epoch, x_max, x_min, xaxis_autorange, auto_range = parse_date_dange(range_d)

        activity_i = interpolate_list(activity)
        # print("activity in spec", activity)
        if len(activity_i) > 1 and None not in activity_i:
            if activity_i is not None and window_size is not None:
                if int(window_size) > len(activity_i):
                    window_size = int(len(activity_i))
            w = signal.blackman(int(window_size))
            f, t, Sxx = signal.spectrogram(np.asarray(activity_i), window=w)
            scales = np.arange(1, int(window_size))
            # scales = even_list(int(window_size))

            sampling_frequency = 1

            if resolution == 'resolution_day':
                sampling_frequency = 86400

            if resolution == 'resolution_hour':
                sampling_frequency = 3600

            if resolution == 'resolution_min':
                sampling_frequency = 60

            if resolution == 'resolution_10min':
                sampling_frequency = 600

            sampling_period = 1 / sampling_frequency
            cwtmatr, freqs = pywt.cwt(np.asarray(activity_i), scales, wavelet, sampling_period=sampling_period)

            cwtmatr_d, freqs_d = pywt.cwt(np.asarray(compute_derivative(activity_i, resolution)), scales, wavelet, sampling_period=sampling_period)

            # plt.pcolormesh(time, freqs, cwtmatr)
            # plt.savefig('spectrum.png')

            transform = "CWT"
            if radio == "STFT":
                transform = Sxx
            if radio == "CWT":
                transform = cwtmatr

            if idx == 1:
                traces.append(go.Heatmap(
                    x=time,
                    y=[-x for x in freqs],
                    z=transform,
                    colorscale='Viridis',
                    colorbar=dict(x=1)
                ))

                traces_derivative.append(go.Heatmap(
                    x=time,
                    y=[-x for x in freqs],
                    z=cwtmatr_d,
                    colorscale='Viridis',
                    colorbar=dict(x=1)
                ))
            if idx == 0:
                if len(data) == 1:
                    d = {}
                else:
                    d = dict(x=1.1)
                traces.append(go.Heatmap(
                    x=time,
                    y=freqs,
                    z=transform,
                    colorbar=d,
                    colorscale='Viridis',
                ))

                traces_derivative.append(go.Heatmap(
                    x=time,
                    y=freqs,
                    z=cwtmatr_d,
                    colorbar=d,
                    colorscale='Viridis'
                ))
        else:
            traces.append({})
            traces_derivative.append({})

    if x_max is not None:
        x_axis_data = {'range': [range_d['x_min'], range_d['x_max']], 'autorange': range_d['xaxis_autorange']}
    else:
        x_axis_data = {'autorange': True}

    q_3.put([{'thread_spectrogram': True}, {'activity': activity_i}, {'time': time}, {
        'data': traces,
        'layout': go.Layout(
            xaxis=x_axis_data,
            yaxis=dict(title='Frequency (Hz)'),
            # autosize=range_d['auto_range'],
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True, legend=dict(y=0.98, x=0.1),
            margin=go.layout.Margin(l=60, r=50, t=5, b=40)),
        'data_derivative': traces_derivative
    }])


def connect_to_sql_database(db_server_name="localhost", db_user="axel", db_password="Mojjo@2015", db_name="",
                            char_set="utf8mb4", cusror_type=pymysql.cursors.DictCursor):
    # print("connecting to db %s..." % db_name)
    global sql_db
    sql_db = pymysql.connect(host=db_server_name, user=db_user, password=db_password,
                             db=db_name, charset=char_set, cursorclass=cusror_type)
    return sql_db


def execute_sql_query(query, records=None, log_enabled=False):
    print(query)
    try:
        sql_db = connect_to_sql_database(db_name=db_name)
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


def get_figure_width(fig_id="activity-graph", url="http://127.0.0.1:8050/"):
    r = requests.get(url)
    # html_string = r.content
    # print(html_string)
    # parsed_html = BeautifulSoup(html_string, 'html.parser')
    # data = parsed_html.find(id=fig_id)
    return -1


def bin_data_to_screen_size(activity_list, timestamp_list, width):
    # print("screen width is %d. There are %d points in activity list." % (width, len(activity_list)))
    n_samples = len(activity_list)
    n_timestamps_per_pixel = n_samples / width
    binned_activity_list = [None for _ in range(width+1)]
    binned_timestamp_list = [None for _ in range(width+1)]
    try:
        for i in xrange(0, len(activity_list)):
            binned_idx = round(i / n_timestamps_per_pixel)
            binned_timestamp_list[binned_idx] = timestamp_list[i]
            if activity_list[i] is None:
                continue
            # need to initialize value for later increment
            if binned_activity_list[binned_idx] is None:
                binned_activity_list[binned_idx] = 0
            binned_activity_list[binned_idx] += activity_list[i]
    except IndexError as e:
        pass
        # print('error wwhile binning data.', e, binned_idx, len(binned_timestamp_list))
    # print("length after scale to screen of width %d is %d." % (width, len(binned_activity_list)))
    # print(binned_activity_list)
    # print(binned_timestamp_list)

    if None in binned_timestamp_list:
        return activity_list, timestamp_list

    # print(activity_list[-1], binned_activity_list[-1])
    return binned_activity_list, binned_timestamp_list


def build_dashboard_layout():
    return html.Div([
        html.Div([html.Pre(id='relayout-data-last-config', style={'display': 'none'})]),
        html.Div(id='output'),
        # Hidden div inside the app that stores the intermediate value
        html.Div(id='intermediate-value', style={'display': 'none'}),
        html.Div(id='activity-mean-value', style={'display': 'none'}),
        html.Div(id='figure-data', style={'display': 'none'}),
        html.Div(id='figure-data-herd', style={'display': 'none'}),
        html.Div(id='figure-data-spectogram', style={'display': 'none'}),
        html.Div(id='dropdown-data-serial', style={'display': 'none'}),
        html.Img(id='logo', style={'max-width': '10%', 'min-width': '10%'},
                 src='http://dof4zo1o53v4w.cloudfront.net/s3fs-public/styles/logo/public/logos/university-of-bristol'
                     '-logo.png?itok=V80d7RFe'),
        html.Br(),
        html.Big(
            children="PhD Thesis: Deep learning of activity monitoring data for disease detection to support "
                     "livestock farming in resource-poor communities in Africa."),
        html.Br(),
        html.Br(),
        # html.B(id='farm-title'),
        html.Div([html.Pre(id='relayout-data', style={'display': 'none'})]),

        html.Div([
            html.Div([
                html.Div([
                    html.Div([html.Label('Farm selection:', style={'color': 'white', 'font-weight': 'bold'}),
                              dcc.Dropdown(
                                  id='farm-dropdown',
                                  options=farm_array,
                                  placeholder="Select farm...",
                                  style={'width': '350px'}
                                  # value=40121100718
                              )],
                             style={'max-width': '350px', 'min-width': '350px', 'width': '350px', 'margin-left': '20px',
                                    'display': 'inline-block'}),

                    html.Div([html.Label('Animal selection:', style={'color': 'white', 'font-weight': 'bold'}),
                              dcc.Dropdown(
                                  id='serial-number-dropdown',
                                  options=[{'label': 'empty', 'value': 'empty'}],
                                  multi=True,
                                  placeholder="Select animal...",
                                  style={'width': '350px', 'margin-bottom': '20px'}
                                  # value=40121100718
                              )],
                             style={'max-width': '350px', 'min-width': '350px', 'width': '350px', 'margin-left': '20px',
                                    'display': 'inline-block'})
                ], style={'float': 'left'}),

                html.Div([

                    html.Div([
                        html.Label('FAMACHA:',
                                   style={'min-width': '100px', 'margin-left': '0vh', 'color': 'white',
                                          'font-weight': 'bold'}),
                        dcc.Checklist(
                            id='cubic-interpolation',
                            options=[
                                {'label': 'Enabled', 'value': 'cubic'}
                            ],
                            value=['cubic'],
                            style={'margin-top': '-50px', 'height': '20px', 'min-width': '100px', 'margin-left': '0px',
                                   'color': 'white',
                                   'font-weight': 'bold', 'display': 'inline-block'}
                        )
                    ],
                        style={'margin-bottom': '30px', 'margin-left': '15px', 'min-width': '120px',
                               'display': 'inline-block'}
                    ),

                    html.Div([
                        html.Label('Normalize:',
                                   style={'min-width': '100px', 'margin-left': '0vh', 'color': 'white',
                                          'font-weight': 'bold'}),
                        dcc.Checklist(
                            id='normalize',
                            options=[
                                {'label': 'Anscombe', 'value': 'Anscombe'},
                                {'label': 'HMean', 'value': 'HMeanDiff'}
                            ],
                            value=['HMeanDiff', 'Anscombe'],
                            labelStyle={'display': 'inline-block'},
                            style={'margin-top': '-50px', 'margin-left': '0px',
                                   'color': 'white',
                                   'font-weight': 'bold', 'display': 'inline-block'}
                        )
                    ],
                        style={'margin-bottom': '30px', 'margin-left': '0px', 'min-width': '120px',
                               'display': 'inline-block'}
                    ),

                    html.Div([
                        html.Label('Transform:', style={'min-width': '100px', 'color': 'white', 'font-weight': 'bold'}),
                        dcc.RadioItems(
                            id='transform-radio',
                            options=[
                                {'label': 'STFT', 'value': 'STFT'},
                                {'label': 'CWT', 'value': 'CWT'}
                            ],
                            labelStyle={'display': 'inline-block', 'color': 'white'},
                            value='CWT')],
                        style={'margin-bottom': '30px', 'margin-left': '10px', 'min-width': '120px',
                               'display': 'inline-block'}
                    ),
                    html.Div([
                        html.Label('Scales:', style={'min-width': '100px', 'margin-left': '0vh', 'color': 'white',
                                                     'font-weight': 'bold'}),
                        dcc.Input(
                            id='window-size-input',
                            placeholder='Input size of window here...',
                            type='text',
                            value='40',
                            style={'min-width': '50px', 'max-width': '50px', 'height': '20px', 'margin-left': '0vh'}
                        )],
                        style={'margin-bottom': '30px', 'margin-left': '10px', 'min-width': '120px',
                               'display': 'inline-block'}
                    ),
                    html.Div([
                        html.Label('Histogram bin:', style={'min-width': '100px', 'margin-left': '0vh', 'color': 'white',
                                                            'font-weight': 'bold'}),
                        dcc.Input(
                            id='histogram-bin-input',
                            placeholder='Input size of window here...',
                            type='text',
                            value='5',
                            style={'min-width': '50px', 'max-width': '50px', 'height': '20px', 'margin-left': '0vh'}
                        )],
                        style={'margin-bottom': '30px', 'margin-left': '-40px', 'min-width': '120px',
                               'display': 'inline-block'}
                    ),

                    html.Div([
                        html.Label('Temperature:',
                                   style={'min-width': '100px', 'margin-left': '0vh', 'color': 'white',
                                          'font-weight': 'bold'}),
                        dcc.Checklist(
                            id='temperature',
                            options=[
                                {'label': 'Enabled', 'value': 'enabled'}
                            ],
                            value=['enabled'],
                            style={'margin-top': '-50px', 'height': '20px', 'min-width': '100px', 'margin-left': '0px',
                                   'color': 'white',
                                   'font-weight': 'bold', 'display': 'inline-block'}
                        )
                    ],
                        style={'margin-bottom': '30px', 'margin-left': '15px', 'min-width': '120px',
                               'display': 'inline-block'}
                    ),

                    html.Div([
                        html.Label('Humidity:',
                                   style={'min-width': '100px', 'margin-left': '0vh', 'color': 'white',
                                          'font-weight': 'bold'}),
                        dcc.Checklist(
                            id='humidity',
                            options=[
                                {'label': 'Enabled', 'value': 'enabled'}
                            ],
                            value=['enabled'],
                            style={'margin-top': '-50px', 'height': '20px', 'min-width': '100px', 'margin-left': '0px',
                                   'color': 'white',
                                   'font-weight': 'bold', 'display': 'inline-block'}
                        )
                    ],
                        style={'margin-bottom': '30px', 'margin-left': '-20px', 'min-width': '80px',
                               'display': 'inline-block'}
                    ),

                    html.Div([
                        html.Label('Ss:',
                                   style={'min-width': '10px', 'margin-left': '0vh', 'color': 'white',
                                          'font-weight': 'bold'}),
                        dcc.Checklist(
                            id='signal_strength',
                            options=[
                                {'label': 'on', 'value': 'enabled'}
                            ],
                            value=['enabled'],
                            style={'margin-top': '-50px', 'height': '20px', 'min-width': '10px', 'margin-left': '0px',
                                   'color': 'white',
                                   'font-weight': 'bold', 'display': 'inline-block'}
                        )
                    ],
                        style={'margin-bottom': '30px', 'margin-left': '0px', 'min-width': '10px',
                               'display': 'inline-block'}
                    ),

                    html.Div([
                        html.Label('Wavelet:',
                                   style={'min-width': '100px', 'margin-left': '0vh', 'color': 'white',
                                          'font-weight': 'bold'}),
                        dcc.RadioItems(
                            id='wavelet-radio',
                            labelStyle={'display': 'inline-block'},
                            options=[
                                {'label': 'mexh', 'value': 'mexh'},
                                {'label': 'morl', 'value': 'morl'},
                                {'label': 'gaus', 'value': 'gaus8'}
                            ],
                            value='mexh',
                            style={'margin-top': '-50px', 'height': '20px', 'min-width': '100px', 'margin-left': '0px',
                                   'color': 'white',
                                   'font-weight': 'bold', 'display': 'inline-block'}
                        )],
                        style={'margin-bottom': '30px', 'margin-left': '0px', 'min-width': '120px',
                               'display': 'inline-block'}
                    ),

                    dcc.Graph(
                        figure=go.Figure(
                            data=[
                                go.Scatter(
                                    x=[1, 2, 3, 4, 5],
                                    y=[1, 2, 3, 4, 5],
                                    name='',
                                    mode='lines',
                                    line=dict(
                                        color=('rgb(255, 255, 255)'))
                                )
                            ],
                            layout=go.Layout(
                                margin=go.layout.Margin(l=0, r=0, t=0, b=0),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                xaxis=dict(showgrid=False),
                                yaxis=dict(showgrid=False),
                            )
                        ),
                        style={'margin-top': '6px', 'margin-right': '100px', 'margin-left': '10px', 'height': '40px', 'width': '80px',
                               'visibility': 'hidden', 'position': 'absolute',
                               'float': 'right'},
                        id='wavelet-graph',
                        config={
                            'displayModeBar': False
                        }
                    )
                ],
                    style={'margin-left': '20px', 'display': 'inline-block', 'min-width': '1050px', 'max-width': '1050px',
                           'margin-bottom': '0px', 'background-color': 'red'}
                )
            ], style={'width': '2520px', 'height': '70px', 'display': 'inline-block', 'background-color': 'gray'})
        ], style={'width': '100%', 'background-color': 'gray'}),



        html.Div([
            html.Div([
                html.Div([
                    # html.Label('logs:'),
                    html.Div([
                        html.Label(
                            "No farm selected.",
                            style={'color': 'white', 'padding-left': '5px', 'font-weight': 'bold'}
                        )
                    ]
                        , id="no-farm-label"
                    ),
                    html.Div(id='log-div', style={'color': 'white'}),
                ], style={'margin-left': '20px', 'background-color': 'gray', 'margin-top': '-5px', 'height': '60px', 'width': '900px', 'display': 'inline-block'})
            ], id='dashboard',
                style={'width': '1920px', 'height': '100px', 'min-height': '50px',
                       'max-height': '50px', 'background-color': 'gray', 'padding-left': '0px',  'padding-bottom': '10px', 'padding-top': '5px'})],
            style={'width': '100%', 'background-color': 'gray'}),





    ], style={'max-height': '350px', 'min-height': '350px', 'background-color': 'white', 'margin-bottom': '0px'})


def get_side_by_side_div(div_l, div_r, offset, height=150, h_div=400, margin_top=0):
    return html.Div([
        html.Div([div_l], style={'height': '%dpx' % height, 'width': '950px', 'float': 'left'}),
        html.Div([div_r], style={'height': str(height+offset)+'px', 'width': '900px', 'float': 'right', 'margin-right': '50px'})
    ],
        style={'height': str(h_div+offset)+'px', 'width': '1920px', 'margin-bottom': '0px', 'margin-top': '%dpx' % margin_top})


def build_derivative_layout():
    return html.Div([
        get_side_by_side_div(
            dcc.Graph(
                figure=go.Figure(
                    data=[
                        go.Scatter(
                            x=[],
                            y=[],
                            name='',
                            mode='lines',
                        )
                    ],
                    layout=go.Layout(
                        margin=go.layout.Margin(l=0, r=0, t=0, b=0)
                    )
                ),
                style={'padding-top': '0vh', 'visibility': 'hidden', 'height': '20vh'},
                id='derivative-graph-spectogram'
            ), dcc.Graph(
                figure=go.Figure(
                    data=[
                        go.Scatter(
                            x=[],
                            y=[],
                            name='',
                            mode='lines',
                        )
                    ],
                    layout=go.Layout(
                        margin=go.layout.Margin(l=0, r=0, t=0, b=0)
                    )
                ),
                style={'padding-top': '0vh', 'visibility': 'hidden', 'height': '20vh'},
                id='derivative-graph'
            ),
            100)

    ], style={'background-color': 'white', 'margin-top': '-140px', 'max-width': '1920px', 'min-width': '1920px',
              'visibility': 'hidden'})


def build_histogram_layout():
    return html.Div([
        dcc.Graph(
            figure=go.Figure(
                data=[
                    go.Scatter(
                        x=[1, 2, 3, 4, 5],
                        y=[1, 2, 3, 4, 5],
                        name='',
                        mode='lines',
                    )
                ],
                layout=go.Layout(
                    margin=go.layout.Margin(l=0, r=0, t=0, b=0)
                )
            ),
            style={'padding-top': '0vh', 'visibility': 'hidden', 'height': '200px' },
            id='histogram-graph'
        )
    ], style={'background-color': 'white', 'margin-top': '-560px', 'max-width': '1920px', 'min-width': '1920px'})


def build_graphs_layout():
    return html.Div([

        get_side_by_side_div(
            dcc.Graph(
                figure=go.Figure(
                    data=[
                        go.Scatter(
                            x=[],
                            y=[],
                            name='',
                        )
                    ],
                    layout=go.Layout(
                        margin=go.layout.Margin(l=0, r=0, t=0, b=0),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                    )
                ),
                style={'margin-top': '-100px', 'height': '600px', 'visibility': 'hidden'},
                id='activity-graph-herd'
                # config={
                #     'displayModeBar': True
                # }

            ), dcc.Graph(
                figure=go.Figure(
                    data=[
                        go.Scatter(
                            x=[],
                            y=[],
                            name='',
                        )
                    ],
                    layout=go.Layout(
                        margin=go.layout.Margin(l=0, r=0, t=0, b=0)
                    )
                ),
                style={'padding-top': '0vh', 'visibility': 'hidden'},
                id='activity-graph'
            ),
            100)
        ,

        get_side_by_side_div(
            dcc.Graph(
                figure=go.Figure(
                    data=[
                        go.Heatmap(
                            x=[],
                            y=[],
                            name='',
                        )
                    ],
                    layout=go.Layout(
                        margin=go.layout.Margin(l=40, r=50, t=50, b=35)

                    )
                ),
                style={'visibility': 'hidden', 'z-index': 0, 'height': '20vh'},
                id='spectrogram-activity-graph'
            ),

            dcc.Graph(
                figure=go.Figure(
                    data=[
                        go.Scatter(
                            x=[],
                            y=[],
                            name='',
                        )
                    ],
                    layout=go.Layout(
                        margin=go.layout.Margin(l=40, r=50, t=50, b=0)
                    )
                ),
                style={'z-index': 1, 'height': '20vh', 'visibility': 'hidden'},
                id='signal-strength-graph'
            )
            , 0, margin_top=-50)

    ], style={'background-color': 'white', 'margin-top': '-40px', 'max-width': '1920px', 'min-width': '1920px'})


def build_default_app_layout(app):
    app.layout = html.Div(
        [
            build_dashboard_layout(), build_graphs_layout(), build_derivative_layout(), build_histogram_layout()
        ], id='main-div')


def check_dragmode(layout_dict_list):
    for layout_dict in layout_dict_list:
        if layout_dict is None:
            continue
        if 'dragmode' in layout_dict:
            return True
    return False


if __name__ == '__main__':
    print("dash ccv %s" % dcc.__version__)
    print(sys.argv)
    q_1 = Queue()
    q_2 = Queue()
    q_3 = Queue()
    q_4 = Queue()
    con = False
    farm_array = []

    if 'sql' == 'h5':
        h5_files_in_data_directory = glob.glob("\*.h5")
        json_files_in_data_directory = glob.glob("\*.json")
        print(h5_files_in_data_directory)
        print(json_files_in_data_directory)
        for s in h5_files_in_data_directory:
            split = s.split("\\")
            farm_name = split[len(split) - 1]
            farm_array.append({'label': str(farm_name), 'value': farm_name})

    if 'sql' == 'sql':
        db_server_name = "localhost"
        db_user = "axel"
        db_password = "Mojjo@2015"
        char_set = "utf8mb4"
        cusror_type = pymysql.cursors.DictCursor

        sql_db = pymysql.connect(host=db_server_name, user=db_user, password=db_password)
        connect_to_sql_database(db_server_name, db_user, db_password, db_name, char_set, cusror_type)
        tables = execute_sql_query("SHOW TABLES", log_enabled=True)
        farm_names = []
        for table in tables:
            name = table["Tables_in_%s" % db_name].split("_")
            farm_names.append("%s_%s" % (name[0], name[1]))
        farm_names = list(set(farm_names))

        for farm_name in farm_names:
            farm_array.append({'label': str(farm_name), 'value': farm_name})

    print('init dash...')
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
    app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/brPBPO.css"})
    build_default_app_layout(app)
    server = app.server

    @app.callback(
        dash.dependencies.Output('relayout-data-last-config', 'children'),
        [Input('relayout-data', 'children')
         ])
    def display_selected_data(v1):
        if v1 is not None:
            d = json.loads(v1)
            if 'dragmode' in d and d['dragmode'] == 'pan':
                print('do not update field!')
                # raise Exception("hack for skipping text field update...")
                raise dash.exceptions.PreventUpdate()
            else:
                return json.dumps(v1, indent=2)

    @app.callback(
        Output('relayout-data', 'children'),
        [Input('activity-graph', 'relayoutData'),
         Input('signal-strength-graph', 'relayoutData'),
         Input('spectrogram-activity-graph', 'relayoutData'),
         Input('activity-graph-herd', 'relayoutData')
         ],
        [dash.dependencies.State('relayout-data-last-config', 'children')]
    )
    def display_selected_data(v1, v2, v3, v5, v4):
        new_layout_data = json.dumps({'autosize': True}, indent=2)

        if check_dragmode([v1, v2, v3, v5]):
            # print("get previous zoom")
            v4 = v4.rstrip()
            last = json.loads(v4).rstrip()
            last = json.loads(last)
            # print("last config:", last)
            last['dragmode'] = "pan"
            new_layout_data = json.dumps(last, indent=2)
            return new_layout_data

        if v1 is not None:
            if "autosize" not in v1 and "xaxis.autorange" not in v1:
                new_layout_data = json.dumps(v1, indent=2)
        if v2 is not None:
            if "autosize" not in v2 and "xaxis.autorange" not in v2:
                new_layout_data = json.dumps(v2, indent=2)
        if v3 is not None:
            if "autosize" not in v3 and "xaxis.autorange" not in v3:
                new_layout_data = json.dumps(v3, indent=2)
        if v5 is not None:
            if "autosize" not in v5 and "xaxis.autorange" not in v5:
                new_layout_data = json.dumps(v5, indent=2)

        # print("new layout data is:", new_layout_data)

        return new_layout_data


    @app.callback(Output('log-div', 'children'),
                  [Input('figure-data', 'children'),
                   Input('serial-number-dropdown', 'options'),
                   Input('farm-dropdown', 'value')])
    def clean_data(data, serial, farm):

        j = json.dumps(serial)
        individual_with_famacha_data = str(j).count('(famacha data available)')
        d = []
        if data is not None:
            d = json.loads(data)
        for i in d:
            signal_size = i['signal_size']
            max_activity_value = i['max_activity_value']
            min_activity_value = i['min_activity_value']
            start_date = i['start_date']
            end_date = i['end_date']
            time_range = i['time_range']
            resolution = i['resolution']

            return html.Div([
                html.Div([html.P("Number of points in signal: %d" % signal_size),
                          html.P("Max activity value: %d" % max_activity_value)], style={'margin-right': '20px', 'display': 'inline-block'}),
                html.Div([html.P("Min activity value: %d" % min_activity_value),
                          html.P("Start date: %s" % start_date)], style={'margin-right': '20px', 'display': 'inline-block'}),
                html.Div([html.P("End date: %s" % end_date),
                          html.P("Time range: %s" % time_range)], style={'margin-right': '20px', 'display': 'inline-block'}),
                html.Div([html.P("Table: %s_%s" % (farm, resolution)),
                          html.P("Table: %s_%s" % (farm, resolution), style={'visibility': 'hidden'})
                          ], style={'display': 'inline-block'}),
            ], style={"width": "1920px"})

        if not d and farm is not None:
            return html.Div([
                html.P("Individual in the farm: %d" % len(serial)),
                html.P("Individual with famacha data available: %s" % individual_with_famacha_data)
            ])


    @app.callback(Output('intermediate-value', 'children'),
                  [Input('farm-dropdown', 'value')])
    def clean_data(farm_id):
        if farm_id is not None:
            print("saving data in hidden div...")
            path = farm_id
            if 'sql' == 'h5':
                h5 = tables.open_file(path, "r")
                serial_numbers = list(set([(x['serial_number']) for x in h5.root.resolution_month.data.iterrows()]))
                print(serial_numbers)
                print("getting data in file...")

                # sort by serial numbers containing data size
                map = {}
                for serial_number in serial_numbers:
                    map[serial_number] = len([x['signal_strength_min'] for x in h5.root.resolution_hour.data if
                                              x['serial_number'] == serial_number])

            if 'sql' == 'sql':
                serial_numbers_rows = execute_sql_query("SELECT DISTINCT(serial_number) FROM %s_resolution_month" % farm_id)
                serial_numbers = [x['serial_number'] for x in serial_numbers_rows]
                print("getting data in file...")
                map = {}
                for serial_number in serial_numbers:
                    rows = execute_sql_query(
                        "SELECT * FROM %s_resolution_week WHERE serial_number=%s" % (farm_id, serial_number),
                        log_enabled=False)
                    map[serial_number] = len(rows)

            sorted_map = sorted(map.items(), key=operator.itemgetter(1))
            sorted_serial_numbers = []
            for item in sorted_map:
                sorted_serial_numbers.append(item[0])

            sorted_serial_numbers.reverse()
            f_id = farm_id.split('.')[0]
            path_json = f_id + "_famacha_data.json"
            famacha_data = {}

            try:
                with open(os.path.join(__location__, path_json)) as f:
                    famacha_data = json.load(f)
            except FileNotFoundError as e:
                print(e)
                print(os.path.join(__location__, path_json))

            path_json_weather = f_id.split('_')[0] + "_weather.json"
            weather_data = {}
            try:
                with open(os.path.join(__location__, path_json_weather)) as f_w:
                    weather_data = json.load(f_w)
            except FileNotFoundError as e:
                print(e)
                os.path.join(__location__, path_json_weather)

            data = {'serial_numbers': sorted_serial_numbers, 'file_path': path, 'farm_id': farm_id,
                    'famacha': famacha_data, 'weather': weather_data}
            return json.dumps(data)


    @app.callback(
        Output('serial-number-dropdown', 'options'),
        [Input('intermediate-value', 'children'),
         Input('dropdown-data-serial', 'children')])
    def update_serial_number_drop_down(intermediate_value, dropdown_value):
        if intermediate_value:
            l = json.loads(intermediate_value)
            data = l["serial_numbers"]
            keys = l['famacha'].keys()
            famacha = list(map(int, keys))
            print("famacha available for:", famacha)
            s_array = []
            for serial in data:
                if dropdown_value is not None:
                    d = json.loads(dropdown_value)
                    if len(d) == 2:
                        if int(serial) not in d:
                            continue

                if serial in famacha:
                    s_array.append({'label': "%s (famacha data available)" % str(serial), 'value': serial})
                else:
                    if serial == 50000000000:
                        s_array.append({'label': 'Mean of herd', 'value': serial})

                    if serial == 60000000000:
                        s_array.append({'label': 'Median of herd', 'value': serial})

                    if serial != 60000000000 and serial != 50000000000:
                        s_array.append({'label': str(serial), 'value': serial})
            return s_array
        else:
            return [{'label': 'empty', 'value': 'empty'}]

    @app.callback(
        Output('figure-data', 'children'),
        [Input('serial-number-dropdown', 'value'),
         Input('intermediate-value', 'children'),
         Input('normalize', 'value'),
         Input('cubic-interpolation', 'value'),
         Input('relayout-data', 'children'),
         Input('activity-mean-value', 'children')
         ])
    def update_figure(selected_serial_number, intermediate_value, normalize, cubic_interp, relayout_data, activity_mean):
        if intermediate_value is not None:
            global sql_db
            print("start thread_activity...")
            p = Process(target=thread_activity,
                        args=(q_1, selected_serial_number, intermediate_value, normalize, cubic_interp, relayout_data, activity_mean))
            p.start()
            result = q_1.get()
            p.join()
            print("thread_activity finished.")
            if len(result) > 0:
                return json.dumps(result, cls=plotly.utils.PlotlyJSONEncoder)


    @app.callback(
        Output('activity-mean-value', 'children'),
        [Input('farm-dropdown', 'value'),
         Input('relayout-data', 'children'),
         Input('normalize', 'value')])
    def update_figure(farm_id, relayout_data, normalize):
        print(farm_id)

        if farm_id is not None:
            range_d = get_date_range(json.loads(relayout_data))
            resolution_string = 'resolution_day'
            if 'x_max_epoch' in range_d:
                x_max_epoch = range_d['x_max_epoch']
                x_min_epoch = range_d['x_min_epoch']
                if range_d['x_max'] is not None:
                    value = find_appropriate_resolution(get_elapsed_time_seconds(x_min_epoch, x_max_epoch))
                    resolution_string = get_resolution_string(value)

                rows_mean = execute_sql_query(
                    "SELECT timestamp, first_sensor_value FROM %s_%s WHERE serial_number=%s AND timestamp BETWEEN %s AND %s" %
                    (farm_id, resolution_string, 50000000000, x_min_epoch, x_max_epoch))
            else:
                rows_mean = execute_sql_query(
                    "SELECT timestamp, first_sensor_value FROM %s_%s WHERE serial_number=%s" %
                    (farm_id, resolution_string, 50000000000))

            activity_m = [(x['first_sensor_value']) for x in rows_mean]
            # if 'Anscombe' in normalize:
            #     activity_m = normalize_activity_array_anscomb(activity_m)

            if len(activity_m) > 0:
                return json.dumps(activity_m, cls=plotly.utils.PlotlyJSONEncoder)


    @app.callback(
        Output('figure-data-herd', 'children'),
        [Input('intermediate-value', 'children'),
         Input('cubic-interpolation', 'value'),
         Input('serial-number-dropdown', 'value'),
         Input('relayout-data', 'children'),
         Input('normalize', 'value'),
         Input('activity-mean-value', 'children')])
    def update_figure(intermediate_value, cubic_interp, selected_seial_number, relayout_data, normalize, activity_mean):
        global sql_db
        print("start thread_activity_herd...")
        p = Process(target=thread_activity_herd,
                    args=(q_4, intermediate_value, cubic_interp, relayout_data, selected_seial_number, normalize, activity_mean,))
        p.start()
        result = q_4.get()
        p.join()
        print("thread_activity_herd finished")
        if len(result) > 0:
            return json.dumps(result, cls=plotly.utils.PlotlyJSONEncoder)


    @app.callback(
        Output('histogram-graph', 'figure'),
        [Input('figure-data', 'children'),
         Input('histogram-bin-input', 'value')
         ])
    def update_figure(data, b):
        if data is not None:
            traces = []
            layout = []
            for idx, d in enumerate(json.loads(data)):
                trace, layout = build_histogram_graph(d['activity'], idx=idx, bin=int(b) if int(b) != 0 else 1)
                traces.append(trace)

            result = {
                'data': traces,
                'layout': layout
            }
            return result
        return {}


    @app.callback(
        Output('spectrogram-activity-graph', 'figure'),
        [Input('figure-data-spectogram', 'children')])
    def hide_graph(data):
        if data is not None:
            d = json.loads(data)
            if len(d) > 2:
                result = {
                    'data': d[3]['data'],
                    'layout': d[3]['layout']
                }
                return result
        return {'data': []}


    # @app.callback(
    #     Output('derivative-graph-spectogram', 'figure'),
    #     [Input('figure-data-spectogram', 'children')])
    # def hide_graph(data):
    #     if data is not None:
    #         d = json.loads(data)
    #         if len(d) > 2:
    #             result = {
    #                 'data': d[3]['data_derivative'],
    #                 'layout': d[3]['layout']
    #             }
    #             return result
    #     return {'data': []}
    #
    #
    # @app.callback(
    #     Output('derivative-graph', 'figure'),
    #     [Input('figure-data', 'children')])
    # def update_figure(data):
    #     if data is not None:
    #         traces = []
    #         layout = []
    #         for idx, d in enumerate(json.loads(data)):
    #             trace, layout = build_derivative_graph(d['activity'], d['resolution'], d['time'], idx=idx)
    #             traces.append(trace)
    #
    #         result = {
    #             'data': traces,
    #             'layout': layout
    #         }
    #         return result
    #     return {}


    @app.callback(
        Output('activity-graph', 'figure'),
        [Input('intermediate-value', 'children'),
         Input('figure-data', 'children'),
         Input('activity-graph', 'relayoutData')])
    def update_figure(data_f, data, last):
        _d = []
        if data is not None:
            _d = json.loads(data)
        _d_f = []
        if data_f is not None:
            _d_f = json.loads(data_f)
        if len(_d) == 0:
            return {}

        figure, layout = build_activity_graph(_d, _d_f, last)
        if figure is not None:
            result = {
                'data': figure[11]['data'],
                'layout': go.Layout(figure[11]['layout'])
            }
            return result

        return {}


    # @app.callback(
    #     Output('histogram-graph-herd', 'figure'),
    #     [Input('figure-data-herd', 'children')])
    # def update_figure(data):
    #     result = {}
    #     if data is not None:
    #         j = json.loads(data)
    #         result = {
    #             'data': j[0]['traces_histogram'],
    #             'layout': j[0]['layout_histogram']
    #         }
    #     print("update_figure activity-graph-herd")
    #
    #     return result


    @app.callback(
        Output('activity-graph-herd', 'figure'),
        [Input('figure-data-herd', 'children')])
    def update_figure(data):
        result = {}
        if data is not None:
            j = json.loads(data)
            s = j[0]['relayout_data']
            layout = json.loads(s)
            layout['paper_bgcolor'] = 'rgba(0,0,0,0)'
            layout['plot_bgcolor'] = 'rgba(0,0,0,0)',
            layout['yaxis'] = {"tickfont": {"size": 11}}
            result = {
                'data': j[0]['traces'],
                'layout': layout
            }
        print("update_figure activity-graph-herd")

        return result

    @app.callback(
        Output('figure-data-spectogram', 'children'),
        [Input('figure-data', 'children'),
         Input('cubic-interpolation', 'value'),
         Input('window-size-input', 'value'),
         Input('transform-radio', 'value'),
         Input('wavelet-radio', 'value'),
         ])
    def update_figure(data, interpolation, window_size, radio, wavelet):
        # if 'cubic' in interpolation:
        j = []
        if data is not None:
            j = json.loads(data)
        result = {
            'data': [],
            'layout': go.Layout(xaxis={'autorange': True},
                                yaxis={'autorange': True, 'title': 'Frequency'},
                                autosize=True,
                                legend=dict(y=0.98), margin=go.layout.Margin(l=60, r=50, t=5, b=40))
        }
        print("start thread_spectrogram...")
        p = Process(target=thread_spectrogram, args=(q_3, window_size, radio, wavelet, j,))
        p.start()
        result = q_3.get(timeout=10)
        p.join()
        print("thread_spectrogram finished.")
        return json.dumps(result, cls=plotly.utils.PlotlyJSONEncoder)


    @app.callback(
        Output('signal-strength-graph', 'figure'),
        [Input('serial-number-dropdown', 'value'),
         Input('intermediate-value', 'children'),
         Input('relayout-data', 'children'),
         Input('temperature', 'value'),
         Input('humidity', 'value'),
         Input('signal_strength', 'value')])
    def update_figure(selected_serial_number, intermediate_value, relayout_data, temperature, humidity, signal_strength):
        print("start thread_signal...")
        p = Process(target=thread_signal, args=(q_2, selected_serial_number, intermediate_value, relayout_data, temperature, humidity, signal_strength,))
        p.start()
        result = q_2.get()
        p.join()
        print("thread_signal finished.")
        return result

    @app.callback(
        Output('wavelet-graph', 'figure'),
        [Input('wavelet-radio', 'value')])
    def hide_graph(value):
        y, x = pywt.ContinuousWavelet(value).wavefun()
        return go.Figure(
            data=[
                go.Scatter(
                    x=x,
                    y=y,
                    name='',
                    mode='lines',
                    line=dict(
                        color=('rgb(255, 255, 255)'))
                )
            ],
            layout=go.Layout(
                margin=go.layout.Margin(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, showline=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, showline=False, zeroline=False, showticklabels=False),
            )
        )


    @app.callback(
        Output('wavelet-graph', 'style'),
        [Input('spectrogram-activity-graph', 'figure')])
    def hide_graph(activity_graph):
        if len(activity_graph['data']) == 0:
            return {'margin-top': '6px', 'margin-right': '100px', 'margin-left': '0px', 'height': '40px', 'width': '80px',
                    'visibility': 'hidden', 'float': 'right'}
        else:
            return {'margin-top': '6px', 'margin-right': '130px', 'margin-left': '0px', 'height': '40px', 'width': '80px',
                    'visibility': 'hidden', 'float': 'right'}
        return {'margin-top': '6px', 'margin-right': '130px', 'margin-left': '0px', 'height': '40px', 'width': '80px',
                'visibility': 'hidden', 'float': 'right'}


    @app.callback(
        Output('derivative-graph', 'style'),
        [Input('derivative-graph', 'figure')])
    def hide_graph(derivative_graph):
        if len(derivative_graph) == 0:
            return {'visibility': 'hidden'}
        else:
            return {'visibility': 'hidden'}


    @app.callback(
        Output('derivative-graph-spectogram', 'style'),
        [Input('derivative-graph-spectogram', 'figure')])
    def hide_graph(derivative_graph_herd):
        if 'data' in derivative_graph_herd and len(derivative_graph_herd['data']) == 0:
            return {'visibility': 'hidden'}
        else:
            return {'visibility': 'hidden'}


    @app.callback(
        Output('histogram-graph', 'style'),
        [Input('histogram-graph', 'figure')])
    def hide_graph(histogram_graph):
        if len(histogram_graph) == 0:
            return {'visibility': 'hidden'}
        else:
            return {'visibility': 'visible'}


    # @app.callback(
    #     Output('histogram-graph-herd', 'style'),
    #     [Input('histogram-graph-herd', 'figure')])
    # def hide_graph(histogram_graph_herd):
    #     if len(histogram_graph_herd) == 0:
    #         return {'visibility': 'hidden'}
    #     else:
    #         return {'visibility': 'visible'}


    @app.callback(
        Output('signal-strength-graph', 'style'),
        [Input('signal-strength-graph', 'figure')])
    def hide_graph(signal_strength_graph):
        if len(signal_strength_graph['data']) == 0:
            return {'visibility': 'hidden'}
        else:
            return {'visibility': 'visible'}


    @app.callback(
        Output('activity-graph-herd', 'style'),
        [Input('activity-graph-herd', 'figure')])
    def hide_graph(activity_graph_herd):
        if 'data' not in activity_graph_herd or len(activity_graph_herd['data']) == 0:
            return {'visibility': 'hidden'}
        else:
            return {'visibility': 'visible', 'margin-top': '-100px'}

    @app.callback(
        Output('activity-graph', 'style'),
        [Input('activity-graph', 'figure')])
    def hide_graph(activity_graph):
        if 'data' not in activity_graph or len(activity_graph['data']) == 0:
            return {'visibility': 'hidden'}
        else:
            return {'visibility': 'visible'}


    @app.callback(
        Output('spectrogram-activity-graph', 'style'),
        [Input('spectrogram-activity-graph', 'figure')])
    def hide_graph(activity_graph):
        if len(activity_graph['data']) == 0:
            return {'visibility': 'hidden'}
        else:
            return {'visibility': 'visible'}


    @app.callback(
        Output('no-farm-label', 'style'),
        [Input('farm-dropdown', 'value')])
    def hide_graph(value):
        if value is not None:
            return {'display': 'none'}


    @app.callback(
        Output('dropdown-data-serial', 'children'),
        [Input('serial-number-dropdown', 'value')]
    )
    def disable_dropdown(value):
        if value is not None:
            if len(value) > 2:
                v = [value[0], value[-1]]
                return json.dumps(v, indent=2)
            else:
                return json.dumps(value, indent=2)


    app.run_server(debug=False, use_reloader=False)
