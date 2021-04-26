#
# Author: Axel Montout <axel.montout <a.t> bristol.ac.uk>
#
# Copyright (C) 2020  Biospi Laboratory for Medical Bioinformatics, University of Bristol, UK
#
# This file is part of PredictionOfHelminthsInfection.
#
# PHI is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PHI is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with seaMass.  If not, see <http://www.gnu.org/licenses/>.
#

#%%

import argparse
import glob
import os
import random
import shutil
import warnings
import pandas as pd
from plotly.subplots import make_subplots
import numpy as np
import plotly.graph_objects as go
from utils.Utils import anscombe, create_rec_dir
from utils._anscombe import Anscombe, Log
from utils._cwt import CWT, CWTVisualisation
from utils._normalisation import QuotientNormalizer
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_jitter, stat_summary, theme

from utils.visualisation import plot_time_pca, plot_groups, plot_time_lda


def plot_zeros_distrib(label_series, data_frame_no_norm, graph_outputdir, title="Percentage of zeros in activity per sample"):
    print("plot_zeros_distrib...")
    data = {}
    target_labels = []
    z_prct = []

    for index, row in data_frame_no_norm.iterrows():
        a = row[:-1].values
        label = label_series[row[-1]]

        target_labels.append(label)
        z_prct.append(np.sum(a == np.log(anscombe(0))) / len(a))

        if label not in data.keys():
            data[label] = a
        else:
            data[label] = np.append(data[label], a)
    distrib = {}
    for key, value in data.items():
        zeros_count = np.sum(value == np.log(anscombe(0))) / len(value)
        lcount = np.sum(data_frame_no_norm["target"] == {v: k for k, v in label_series.items()}[key])
        distrib[str(key) + " (%d)" % lcount] = zeros_count

    plt.bar(range(len(distrib)), list(distrib.values()), align='center')
    plt.xticks(range(len(distrib)), list(distrib.keys()))
    plt.title(title)
    plt.xlabel('Famacha samples (number of sample in class)')
    plt.ylabel('Percentage of zero values in samples')
    # plt.show()
    print(distrib)

    df = pd.DataFrame.from_dict({'Percent of zeros': z_prct, 'Target': target_labels})
    df.to_csv(graph_outputdir + "/z_prct_data.csv")
    g = (ggplot(df)  # defining what data to use
         + aes(x='Target', y='Percent of zeros', color='Target', shape='Target')  # defining what variable to use
         + geom_jitter()  # defining the type of plot to use
         + stat_summary(geom="crossbar", color="black", width=0.2)
         + theme(subplots_adjust={'right': 0.82})
         )

    fig = g.draw()
    fig.tight_layout()
    # fig.show()
    filename = "zero_percent_%s.png" % title.lower().replace(" ","_")
    filepath = "%s/%s" % (graph_outputdir, filename)
    # print('saving fig...')
    fig.savefig(filepath)
    # print("saved!")
    fig.clear()
    plt.close(fig)


def plotHeatmap(X, out_dir="", title="Heatmap", filename="heatmap.html", y_log=False):
    # fig = make_subplots(rows=len(transponders), cols=1)
    ticks = list(range(X.shape[1]))
    fig = make_subplots(rows=1, cols=1)
    if y_log:
        X_log = np.log(anscombe(X))
    trace = go.Heatmap(
            z=X_log if y_log else X,
            x=ticks,
            y=list(range(X.shape[0])),
            colorscale='Viridis')
    fig.add_trace(trace, row=1, col=1)
    fig.update_layout(title_text=title)
    fig.update_layout(xaxis_title="Time in minutes")
    #fig.show()
    create_rec_dir(out_dir)
    file_path = out_dir + "/" + filename.replace("=", "_").lower()
    print(file_path)
    fig.write_html(file_path)
    return trace, title


def setupGraphOutputPath(output_dir):
    graph_outputdir = "%s/input_graphs/" % output_dir
    if os.path.exists(graph_outputdir):
        print("purge %s ..." % graph_outputdir)
        try:
            shutil.rmtree(graph_outputdir)
        except IOError:
            print("file not found.")
    create_rec_dir(graph_outputdir)
    return graph_outputdir


def applyPreprocessingSteps(df, N_META, output_dir, steps, class_healthy_label, class_unhealthy_label, class_healthy, class_unhealth):

    step_slug = "_".join(steps)
    graph_outputdir = setupGraphOutputPath(output_dir) + "/" + step_slug

    if len(steps) == 0:
        print("no steps to apply! return data as is")
        return df
    print("BEFORE STEP ->", df)
    for step in steps:

        if step not in ["ANSCOMBE", "LOG", "QN", "CWT"]:
            warnings.warn("processing step %s does not exist!" % step)

        print("applying STEP->%s in [%s]..." % (step, step_slug.replace("_", "->")))
        if step == "ANSCOMBE":
            df.iloc[:, :-N_META] = Anscombe().transform(df.iloc[:, :-N_META].values)
        if step == "LOG":
            df.iloc[:, :-N_META] = Log().transform(df.iloc[:, :-N_META].values)
        if step == "QN":
            df.iloc[:, :-N_META] = QuotientNormalizer(out_dir=graph_outputdir + "/" +step).transform(df.iloc[:, :-N_META].values)
        if step == "CWT":
            df_o = df.copy()
            CWT_Transform = CWT(out_dir=graph_outputdir + "/" + step)
            data_frame_cwt = pd.DataFrame(
                CWT_Transform.transform(df.copy().iloc[:, :-N_META].values))
            data_frame_cwt.index = df.index  # need to keep original sample index!!!!
            df_meta = df.iloc[:, -N_META:]
            df = pd.concat([data_frame_cwt, df_meta], axis=1)
            # sanity check#################################################################################################
            rdm_idxs = random.choices(df.index.tolist(), k=1)
            samples_tocheck = df_o.loc[(rdm_idxs), :].values[:, :-N_META]
            cwt_to_check = pd.DataFrame(CWT(out_dir=graph_outputdir + "/" + step + "/cwt_sanity_check/").transform(samples_tocheck))
            prev_cwt_results = df.loc[(rdm_idxs), :].values[:, :-N_META]
            assert False not in (cwt_to_check.values == prev_cwt_results), "missmatch in cwt sample!"
            #############################################################################################################
            data_frame_cwt_full = pd.DataFrame(CWT_Transform.cwt_full)
            data_frame_cwt_full.index = df.index# need to keep original sample index!!!!
            CWTVisualisation(output_dir, CWT_Transform.shape, CWT_Transform.freqs, CWT_Transform.coi, df_o.copy(),
                             data_frame_cwt_full, class_healthy_label, class_unhealthy_label, class_healthy, class_unhealthy)
        print("AFTER STEP ->", df)
    return df


def loadActivityData(filepath):
    print("load activity from datasets...", filepath)
    data_frame = pd.read_csv(filepath, sep=",", header=None, low_memory=False)
    data_frame = data_frame.astype(dtype=float, errors='ignore')  # cast numeric values as float
    data_point_count = data_frame.shape[1]
    hearder = [str(n) for n in range(0, data_point_count)]
    N_META = 4
    hearder[-4] = 'label'
    hearder[-3] = 'id'
    hearder[-2] = 'imputed_days'
    hearder[-1] = 'date'
    data_frame.columns = hearder
    data_frame = data_frame[~np.isnan(data_frame["imputed_days"])]
    data_frame = data_frame.fillna(-1)
    # filter with imputed_days count
    data_frame = data_frame[data_frame["imputed_days"] >= day]
    return data_frame, N_META


if __name__ == "__main__":
    print("ML PIPELINE")
    print("********************************************************************")
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', help='output directory', type=str)
    parser.add_argument('dataset_folder', help='dataset input directory', type=str)
    parser.add_argument('--class_healthy', help='target for healthy class', default=1, type=int)
    parser.add_argument('--class_unhealthy', help='target for unhealthy class', default=2, type=int)
    parser.add_argument('--stratify', help='enable stratiy for cross validation', default='n', type=str)
    parser.add_argument('--s_output', help='output sample files', default='y', type=str)
    parser.add_argument('--cwt', help='enable freq domain (cwt)', default='y', type=str)
    parser.add_argument('--temp_file', help='temperature features.', default=None, type=str)
    parser.add_argument('--hum_file', help='humidity features.', default=None, type=str)
    parser.add_argument('--n_process', help='number of threads to use.', default=6, type=int)
    args = parser.parse_args()

    output_dir = args.output_dir
    dataset_folder = args.dataset_folder
    class_healthy = args.class_healthy
    class_unhealthy = args.class_unhealthy
    stratify = args.stratify
    s_output = args.s_output
    cwt = args.cwt
    hum_file = args.hum_file
    temp_file = args.temp_file
    n_process = args.n_process

    stratify = "y" in stratify.lower()
    output_samples = "y" in s_output.lower()
    output_cwt = "y" in cwt.lower()

    print("output_dir=", output_dir)
    print("dataset_filepath=", dataset_folder)
    print("class_healthy=", class_healthy)
    print("class_unhealthy=", class_unhealthy)
    print("output_samples=", output_samples)
    print("stratify=", stratify)
    print("output_cwt=", output_cwt)
    print("hum_file=", hum_file)
    print("temp_file=", temp_file)
    print("n_process=", n_process)
    print("loading dataset...")
    enable_downsample_df = False
    day = int(dataset_folder.split('_')[-1][0])

    files = glob.glob(dataset_folder + "/*.csv")  # find datset files
    files = [file.replace("\\", '/') for file in files]
    print("found %d files." % len(files))
    print(files)

    has_humidity_data = False
    if hum_file is not None:
        has_humidity_data = True
        print("humidity file detected!", hum_file)
        df_hum = pd.read_csv(hum_file)
        print(df_hum.shape)
        plotHeatmap(df_hum.values, output_dir, "Samples humidity", "humidity.html")

    has_temperature_data = True
    if temp_file is not None:
        has_temperature_data = True
        print("temperature file detected!", temp_file)
        df_temp = pd.read_csv(temp_file)
        plotHeatmap(df_temp.values, output_dir, "Samples temperature", "temperature.html")
        print(df_temp.shape)

    has_humidity_and_temp = False
    if temp_file is not None and hum_file is not None:
        has_humidity_and_temp = True
        print("temperature file detected!", temp_file)
        print("humidity file detected!", hum_file)
        df_hum_temp = pd.concat([df_temp, df_hum], axis=1)
        plotHeatmap(df_hum_temp.values, output_dir, "Samples temperature and Humidity", "temperature_humidity.html")
        print(df_hum_temp.shape)

    for file in files:
        print("loading dataset file %s ..." % file)
        data_frame, N_META = loadActivityData(file)
        data_frame_o = data_frame.copy()
        print(data_frame)

        # Hot Encode of FAmacha targets and assign integer target to each famacha label
        data_frame_labeled = pd.get_dummies(data_frame, columns=["label"])
        flabels = [x for x in data_frame_labeled.columns if 'label' in x]
        data_frame["target"] = 0
        for i, flabel in enumerate(flabels):
            data_frame_labeled[flabel] = data_frame_labeled[flabel] * (i + 1)
            data_frame["target"] = data_frame["target"] + data_frame_labeled[flabel]
        class_count = {}
        label_series = dict(data_frame[['target', 'label']].drop_duplicates().values)
        print(label_series)
        class_healthy_label = label_series[class_healthy]
        class_unhealthy_label = label_series[class_unhealthy]
        for k in label_series.keys():
            class_count[label_series[k] + "_" + str(k)] = data_frame[data_frame['target'] == k].shape[0]
        print(class_count)
        # drop label column stored previously, just keep target for ml
        data_frame = data_frame.drop('label', 1)
        print(data_frame)
        # keep only two class of samples
        data_frame = data_frame[data_frame["target"].isin([class_healthy, class_unhealthy])]

        # ["QN", "ANSCOMBE", "LOG", "CWT"]
        df_processed = applyPreprocessingSteps(data_frame.copy(), N_META, output_dir, ["CWT"],
                                               class_healthy_label, class_unhealthy_label, class_healthy, class_unhealthy)

        df_processed = df_processed.iloc[:, :-N_META + 1]

        ##VISUALISATION
        df_norm = applyPreprocessingSteps(data_frame.copy(), N_META, output_dir, ["QN"])
        plot_zeros_distrib(label_series, df_norm, output_dir,
                           title='Percentage of zeros in activity per sample after normalisation')
        plot_zeros_distrib(label_series, data_frame.copy(), output_dir,
                           title='Percentage of zeros in activity per sample before normalisation')

        plot_time_pca(data_frame.copy(), output_dir, label_series, title="PCA time domain before normalisation")
        plot_time_pca(df_norm, output_dir, label_series, title="PCA time domain after normalisation")

        plot_time_lda(data_frame.copy(), output_dir, label_series, title="LDA time domain before normalisation")
        plot_time_lda(data_frame.copy(), output_dir, label_series, title="LDA time domain after normalisation")

        animal_ids = df_norm.iloc[0:len(df_norm), :]["id"].astype(str).tolist()
        ntraces = 2
        idx_healthy, idx_unhealthy = plot_groups(animal_ids, class_healthy_label, class_unhealthy_label, class_healthy,
                                                 class_unhealthy, output_dir, data_frame.copy(), title="Raw imputed",
                                                 xlabel="Time",
                                                 ylabel="activity", ntraces=ntraces)
        plot_groups(animal_ids, class_healthy_label, class_unhealthy_label, class_healthy, class_unhealthy,
                    output_dir,
                    df_norm, title="Normalised(Quotient Norm) samples", xlabel="Time", ylabel="activity",
                    idx_healthy=idx_healthy, idx_unhealthy=idx_unhealthy, stepid=2, ntraces=ntraces)


