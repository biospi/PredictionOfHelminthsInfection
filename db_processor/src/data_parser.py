import time
import os
import xlrd
from ipython_genutils.py3compat import xrange


def purge_file(filename):
    print("purge %s..." % filename)
    try:
        os.remove(filename)
    except FileNotFoundError:
        print("file not found.")


def find_data_files(directory_path, extension='.xls'):
    print("start searching for data files...")
    os.chdir(directory_path)
    file_paths = [val for sublist in
                  [[os.path.join(i[0], j) for j in i[2] if j.endswith(extension)] for i in os.walk(directory_path)] for
                  val in sublist]
    print("founded %d %s files" % (len(file_paths), extension))
    return file_paths


def file_contains_famacha(book):
    for n in range(book.nsheets):
        sheet = book.sheet_by_index(n)
        for row_index in xrange(0, sheet.nrows):
            try:
                row_values = [sheet.cell(row_index, col_index).value for col_index in xrange(0, sheet.ncols)]
                if 'Famacha' in row_values:
                    return True
            except Exception as e:
                print(e)
    return False


def process_files(file_paths):
    for curr_file, path in enumerate(file_paths):
        try:
            print("loading file in memory for reading...")
            print(path)
            book = xlrd.open_workbook(path)
            if not file_contains_famacha(book):
                continue

            for n in range(book.nsheets):
                sheet = book.sheet_by_index(n)
                print("start reading...")
                found_col_index = False
                for row_index in xrange(0, sheet.nrows):
                    try:
                        row_values = [sheet.cell(row_index, col_index).value for col_index in xrange(0, sheet.ncols)]
                        print(row_values)
                    except Exception as e:
                        print(e)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    print("start...")
    start_time = time.time()
    xls_files = find_data_files("E:/SouthAfrica/Metadata/BOTHAVILLE data", extension='.xls')
    pdf_files = find_data_files("E:/SouthAfrica/Metadata/BOTHAVILLE data", extension='.pdf')

    process_files(xls_files)

