from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def rmse(ori_x_rmse, imputed_x, data_mask):
    d_m = data_mask.copy()
    diff_ = ori_x_rmse * d_m - imputed_x * d_m
    nominator_ = np.sum(diff_ ** 2)
    denominator_ = np.sum(d_m)
    rmse = np.sqrt(float(nominator_) / float(denominator_))
    #print("nominator=", nominator_)
    #print("denominator=", denominator_)
    return rmse


def process(df_raw_, df_gain_, df_mrnn):
    df_gain_ = df_gain_.dropna()
    df_mrnn = df_mrnn.loc[df_gain_.index]
    df_raw_ = df_raw_.loc[df_gain_.index]

    X_gain = df_gain_.iloc[:, :-4]
    X_mrnn = df_mrnn.iloc[:, :-4]
    X_raw = df_raw_.iloc[:, :-4]
    X_li = X_raw.interpolate(axis=1, limit_direction='both')

    sample_weight = np.zeros(X_raw.shape)
    sample_weight[(np.isnan(X_raw)) & (X_mrnn >0)  & (X_gain >0)] = 1

    X_gain = X_gain.fillna(0)
    X_mrnn = X_mrnn.fillna(0)
    X_raw = X_raw.fillna(0)
    X_li = X_li.fillna(0)

    rmse_mrnn_list = []
    rmse_gain_list = []
    rmse_li_list = []
    for i in range(X_raw.shape[0]):
        rmse_mrnn = mean_squared_error(X_raw.iloc[i,:], X_mrnn.iloc[i,:], sample_weight=sample_weight[i,:], squared=False)
        rmse_mrnn_list.append(rmse_mrnn)
        print(f"raw vs mrnn rmse= {rmse_mrnn}")
        rmse_gain = mean_squared_error(X_raw.iloc[i,:], X_gain.iloc[i,:], sample_weight=sample_weight[i,:], squared=False)
        rmse_gain_list.append(rmse_gain)
        print(f"raw vs gain rmse= {rmse}")
        rmse_li = mean_squared_error(X_raw.iloc[i,:], X_li.iloc[i,:], sample_weight=sample_weight[i,:], squared=False)
        rmse_li_list.append(rmse_li)
        print(f"raw vs li rmse= {rmse_li}")

    df = pd.DataFrame()
    df["RAW vs MRNN"] = rmse_mrnn_list
    df["RAW vs GAIN"] = rmse_gain_list
    df["RAW vs LI"] = rmse_li_list
    print(df)
    fig_ = df.boxplot(column=['RAW vs MRNN', 'RAW vs GAIN', 'RAW vs LI']).get_figure()
    ax = fig_.gca()
    ax.set_title("RMSE comparison for each ML samples")
    ax.set_ylabel("RMSE")
    ax.set_xlabel("Imputation")
    filename = f"RMSE_comparison_ml_samples.png"
    filepath = filename
    print(filepath)
    fig_.tight_layout()
    fig_.savefig(filepath)


def find_samples(df, date, id, label):
    header = df.columns.tolist()
    header[-1] = "date"
    header[-3] = "id"
    header[-4] = "label"
    df.columns = header
    df_sample = df[df["date"] == date]
    df_sample = df_sample[df_sample["label"] == label]
    df_sample = df_sample[df_sample["id"] == id]
    return df_sample.values.flatten()


def main(dataset, dataset_mrnn, dataset_gain):
    print(dataset, dataset_mrnn, dataset_gain)
    df_dataset = pd.read_csv(dataset, header=None)
    df_mrnn = pd.read_csv(dataset_mrnn, header=None)
    df_gain = pd.read_csv(dataset_gain, header=None)
    samples_raw = []
    samples_gain = []
    for index, row in df_mrnn.iterrows():
        date = row.values[-1]
        id = row.values[-3]
        label = row.values[-4]
        sample_raw = find_samples(df_dataset, date, id, label)
        samples_raw.append(sample_raw)
        sample_gain = find_samples(df_gain, date, id, label)
        samples_gain.append(sample_gain)
    df_gain_ = pd.DataFrame(samples_gain)
    df_raw_ = pd.DataFrame(samples_raw)
    process(df_raw_, df_gain_, df_mrnn)


if __name__ == "__main__":
    main(Path("E:/thesis/datasets/delmas/datasetraw_none_7day/activity_farmid_dbft_7_1min.csv"),
         Path("E:/thesis/datasets/delmas/datasetmrnn7/activity_farmid_dbft_7_1min.csv"),
         Path("E:/thesis/datasets/delmas/datasetmrnn7_gain/activity_farmid_dbft_7_1min.csv"))