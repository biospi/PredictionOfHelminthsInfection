import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import shutil
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 50)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)
import statistics

if __name__ == '__main__':
    print("start...")
    try:
        shutil.rmtree("result_analysis")
    except (OSError, FileNotFoundError) as e:
        print(e)

    fname = "E:\\Users\\fo18103\\PycharmProjects\\prediction_of_helminths_infection\\training_data_generator_and_ml_classifier\\src\\delmas_70101200027_results_simplified_report_2020_04_03_03_41_02.csv"
    df = pd.read_csv(fname, sep=",")
    df = df[df.classifier != "empty"]
    df = df[df.sliding_w == 0]
    df["inputs"] = df["inputs"].str.replace("\n", '')
    df = df.sort_values(' accuracy', ascending=False)
    print(df)
    df = df[[' accuracy', 'specificity', 'recall', 'precision', 'fscore', 'days', 'resolution', 'inputs', 'proba_y_false', 'proba_y_true']]
    print(df)
    for index, row in df.iterrows():
        proba_false = [float(x) for x in row['proba_y_false'].strip('()').strip('[]')[:-2].split('- ')]
        proba_true = [float(x) for x in row['proba_y_true'].strip('()').strip('[]')[:-2].split('- ')]
        p_f = statistics.mean(proba_false)
        p_t = statistics.mean(proba_true)
        line = "%.2f&%.2f&%.2f&%.2f&%.2f&%.2f&%.2f&%d&%s&%s\\\\" % (row[' accuracy'], row['specificity'], row['recall'], row['precision'], row['fscore'], p_f, p_t, row['days'], row['resolution'], row['inputs'])
        print(line)
    # list_of_df = [g for _, g in df.groupby(['resolution', 'inputs', 'sliding_w', 'classifier'])]
    list_of_df = []
    list_of_df_filtered = []
    for item in list_of_df:
        if item.shape[0] >= 6:
            item = item.reset_index(drop=True)
            list_of_df_filtered.append(item)
            if item['sliding_w'][0] == 0:
                continue
            print(item)
            fig = plt.figure()
            title = "%s_%s_%s" % (item['resolution'][0], item['inputs'][0], item['classifier'][0])
            plt.title(title)
            plt.plot(item['days'], item[' accuracy'])
            plt.show()
            pathlib.Path("result_analysis").mkdir(parents=True, exist_ok=True)
            path = "result_analysis\\" + title + ".png"
            fig.savefig(path)



