import os
import os.path
import os.path
import sys
import time
from datetime import datetime
from sys import exit

import dateutil.relativedelta
import pandas as pd
import xlrd
from ipython_genutils.py3compat import xrange


def xl_date_to_date(xldate, wb):
    year, month, day, hour, minute, second = xlrd.xldate_as_tuple(xldate, wb.datemode)
    return "%02d/%02d/%d" % (day, month, year)


def get_elapsed_time_string(time_initial, time_next):
    dt1 = datetime.fromtimestamp(time_initial)
    dt2 = datetime.fromtimestamp(time_next)
    rd = dateutil.relativedelta.relativedelta(dt2, dt1)
    return '%02d:%02d:%02d:%02d' % (rd.days, rd.hours, rd.minutes, rd.seconds)


def convert_excel_time(cell_with_excel_time, wb):
    year, month, day, hour, minute, second = xlrd.xldate_as_tuple(cell_with_excel_time, wb.datemode)  # year=month=day=0
    py_date = datetime(year=1989, month=4, day=12, hour=hour, minute=minute,
                       second=second)  # used dummy date we only need time
    time_value = py_date.strftime("%I:%M:%S %p")
    return time_value


def print_except(e):
    template = "An exception of type {0} occurred. Arguments:\n{1!r}"
    message = template.format(type(e).__name__, e.args)
    print(message)


def empty_list(l):
    return len(l) == l.count('')


def strip_list(l):
    # print(l)
    return [x.strip().lower().replace(" ", "").replace("batteryife", "batterylife") for x in l]


def generate_raw_files_from_xlsx(directory_path, file_name):
    start_time = time.time()
    print("start readind xls files...")
    log_file = open("log.txt", "a")
    file_paths = [val for sublist in
                  [[os.path.join(i[0], j) for j in i[2] if j.endswith('.xlsx')] for i in os.walk(directory_path)] for
                  val in sublist]
    print("founded %d files" % len(file_paths))
    print(file_paths)
    print("start generating raw file...")
    store = pd.HDFStore(file_name)
    valid_rows = 0
    for curr_file, path in enumerate(file_paths):
        print('progress %s/%d' % (curr_file, len(file_paths)))
        df = []
        transponders = {}
        try:
            record_log = ""
            print("loading file in memory for reading...")
            print(path)
            book = xlrd.open_workbook(path)
            for sheet in book.sheets():
                print('reading sheet', sheet.name)
                # sheet = book.sheet_by_index(0)
                try:
                    farm_id = int(sheet.name.split('_')[0])
                except ValueError as e:
                    print("error while parsing sheet name")
                    print(e)
                    print(path)

                print("start reading...")
                found_col_index = False
                for row_index in xrange(0, sheet.nrows):
                    # if row_index > 10:
                    #     break
                    try:
                        row_values = [sheet.cell(row_index, col_index).value for col_index in xrange(0, sheet.ncols)]
                        if len(row_values) == 0:
                            continue
                        # print(row_values)

                        if 'Cedara' in directory_path and 'Sender' in path and not found_col_index:
                            date_col_index = 0
                            time_col_index = 1
                            control_station_col_index = 2
                            try:
                                farm_id = int(row_values[control_station_col_index])
                            except ValueError as e:
                                print("error while parsing farm id")
                                print(e)
                            serial_number_col_index = 3
                            signal_strength_col_index = 4
                            battery_voltage_col_index = 5
                            first_sensor_value_col_index = 6
                            found_col_index = True
                            print("found header. parsing rows...")
                            continue

                        if not found_col_index:
                            try:
                                row_values = strip_list(row_values)
                            except AttributeError as e:
                                print_except(e)
                                date_col_index = 0
                                time_col_index = 1
                                control_station_col_index = 2
                                try:
                                    farm_id = int(row_values[control_station_col_index])
                                except ValueError as e:
                                    print(e)
                                except IndexError as e:
                                    print(e)
                                    continue
                                serial_number_col_index = 3
                                signal_strength_col_index = 4
                                battery_voltage_col_index = 5
                                first_sensor_value_col_index = 6
                                found_col_index = True
                                print("found header. parsing rows...")
                                continue
                                # exit()

                        # print(path)
                        # print(row_values)
                        # find index of each column
                        if not found_col_index:
                            try:
                                date_col_index = row_values.index('date')
                                time_col_index = row_values.index('time')
                                control_station_col_index = row_values.index('controlstation')
                                serial_number_col_index = row_values.index('tagserialnumber')
                                signal_strength_col_index = row_values.index('signalstrength')
                                battery_voltage_col_index = row_values.index('batteryvoltage')
                                first_sensor_value_col_index = row_values.index('firstsensorvalue')
                                found_col_index = True

                                print("found header. parsing rows...")
                                continue
                            except ValueError as e:
                                if 'transpondernumber' in row_values and 'activitylevel' in row_values:
                                    date_col_index = row_values.index('date')
                                    time_col_index = row_values.index('time')
                                    serial_number_col_index = row_values.index('transpondernumber')
                                    signal_strength_col_index = row_values.index('signalstrength')
                                    battery_voltage_col_index = row_values.index('batterylife')
                                    first_sensor_value_col_index = row_values.index('activitylevel')
                                    found_col_index = True

                                    print("found header. parsing rows...")
                                    continue
                                if 'transponderno' in row_values and 'activitylevel' in row_values:
                                    date_col_index = row_values.index('date')
                                    time_col_index = row_values.index('time')
                                    serial_number_col_index = row_values.index('transponderno')
                                    first_sensor_value_col_index = row_values.index('activitylevel')
                                    if 'signalstrength' in row_values:
                                        signal_strength_col_index = row_values.index('signalstrength')
                                    if 'batterylife' in row_values:
                                        battery_voltage_col_index = row_values.index('batterylife')
                                    found_col_index = True

                                    print("found header. parsing rows...")
                                    continue
                                continue

                        # print("farm id is %d." % farm_id)
                        # print('progress %s/%d' % (curr_file, len(file_paths)))

                        try:
                            date_string = xl_date_to_date(row_values[date_col_index], book) + " " + convert_excel_time(
                                row_values[time_col_index], book)
                            # print(date_string)
                        except TypeError as e:
                            print("error while parsing date", row_values)
                            print(e)
                            continue

                        epoch = int(datetime.strptime(date_string, '%d/%m/%Y %I:%M:%S %p').timestamp())
                        control_station = farm_id  # int(row_values[control_station_col_index])
                        serial_number = int(row_values[serial_number_col_index])

                        try:
                            signal_strength = int(
                                str(row_values[signal_strength_col_index]).replace("@", "").split('.')[0])
                        except (IndexError, NameError, UnboundLocalError, ValueError) as e:
                            # print(e)
                            # print(path)
                            signal_strength = -1
                        try:
                            battery_voltage = int(str(row_values[battery_voltage_col_index]).split('.')[0], 16)
                        except (IndexError, NameError, UnboundLocalError, ValueError) as e:
                            # print(e)
                            # print(path)
                            battery_voltage = -1

                        try:
                            first_sensor_value = int(row_values[first_sensor_value_col_index])
                        except ValueError as e:
                            print("error while parsing first sensor value")
                            print(e)
                            print(path)
                            continue

                        # print(curr_file, len(file_paths), farm_id, sheet.name, row_values, path,
                        #       first_sensor_value_col_index, battery_voltage_col_index, signal_strength_col_index,
                        #       serial_number_col_index, date_col_index, time_col_index)

                        record_log = "date_string=%s time=%s  row=%d epoch=%d control_station=%d serial_number=%d signal_strength=%d battery_voltage=%d first_sensor_value=%d" % (
                            date_string,
                            get_elapsed_time_string(start_time, time.time()), valid_rows, epoch, control_station,
                            serial_number,
                            signal_strength, battery_voltage, first_sensor_value)
                        # print(record_log)

                        transponders[serial_number] = ''

                        df.append(
                            pd.DataFrame({
                                'timestamp': epoch,
                                'control_station': control_station,
                                'serial_number': serial_number,
                                'signal_strength': signal_strength,
                                'battery_voltage': battery_voltage,
                                'first_sensor_value': first_sensor_value
                            }, index=[valid_rows]))

                        valid_rows += 1
                    except ValueError as exception:
                        print("global error")
                        print_except(exception)
                        print(exception)
                        print(path)
                        if not None:
                            print(row_values)
                        log = "%d/%d--%s---%s---%s---%s" % (
                            curr_file, len(file_paths), get_elapsed_time_string(start_time, time.time()),
                            str(exception), path,
                            record_log)
                        print(log)
                        log_file.write(log + "\n")

                print("transponders:", transponders.keys())
                if len(df) == 0:
                    continue

                store.append('/', value=pd.concat(df), format='t', append=True,
                             data_columns=['timestamp', 'control_station', 'serial_number', 'signal_strength',
                                           'battery_voltage', 'first_sensor_value'])

                # del table_row
                # table_f.flush()
            del book
            del df
        except (ValueError, IOError, xlrd.biffh.XLRDError) as e:
            print('error', e)
            print(path)
            continue

    print("saving %s" % (file_name))
    store.close()
    print("done")


def create_rec_dir(path):
    dir_path = ""
    sub_dirs = path.split("/")
    for sub_dir in sub_dirs[0:]:
        dir_path += sub_dir+"/"
        # print("sub_folder=", dir_path)
        if not os.path.exists(dir_path):
            print("mkdir", dir_path)
            os.makedirs(dir_path)


if __name__ == '__main__':

    print("args: output_filename raw_csv_file_dir")
    if len(sys.argv) > 1:
        output_filename = sys.argv[1]
        raw_csv_file_dir = sys.argv[2]
    else:
        exit(-1)

    out_dir = '/'.join(output_filename.split('/')[:-1])
    print("out_dir=", out_dir)
    create_rec_dir(out_dir)
    print("output_filename=", output_filename)
    print("raw_csv_file_dir=", raw_csv_file_dir)
    generate_raw_files_from_xlsx(raw_csv_file_dir, output_filename)