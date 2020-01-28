import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import shutil
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 50)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

if __name__ == '__main__':
    print("start...")
    try:
        shutil.rmtree("result_analysis")
    except (OSError, FileNotFoundError) as e:
        print(e)

    fname = "E:\\Users\\fo18103\\PycharmProjects\\prediction_of_helminths_infection\\" \
            "training_data_generator_and_ml_classifier\\" \
            "src\\delmas_70101200027_results_simplified_report_2020_01_27_16_29_48.csv"
    df = pd.read_csv(fname, sep=",")
    df = df[df.classifier != "empty"]
    df = df.sort_values('days')
    df["inputs"] = df["inputs"].str.replace("\n", '')
    list_of_df = [g for _, g in df.groupby(['resolution', 'inputs', 'sliding_w', 'classifier'])]
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



