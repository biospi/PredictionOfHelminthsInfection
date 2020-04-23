import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import shutil
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 50)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)
import statistics
import numpy as np

def get_ranged_value(col_name):
    col_data = [float(x) for x in row[col_name].split(' ')]
    mean_value = statistics.mean(col_data)
    plus_minus = np.std(col_data)
    return mean_value, plus_minus


if __name__ == '__main__':
    print("start...")
    try:
        shutil.rmtree("result_analysis")
    except (OSError, FileNotFoundError) as e:
        print(e)

    fname = "E:\\Users\\fo18103\\PycharmProjects\\prediction_of_helminths_infection\\training_data_generator_and_ml_classifier\\src\\sd\\delmas_70101200027_results_report_2020_04_14_15_36_48.csv"
    #fname = "E:\\Users\\fo18103\\PycharmProjects\\prediction_of_helminths_infection\\training_data_generator_and_ml_classifier\\src\\sd\\cedara_70091100056_results_report_2020_04_13_06_41_36.csv"

    df = pd.read_csv(fname, sep=",")
    df.columns = [x.replace(' ', '') for x in df.columns]
    df = df[df.classifier != "empty"]
    df = df[df.sliding_w == 0]
    df["input"] = df["input"].str.replace("\n", '')
    df = df.sort_values('accuracy_cv', ascending=False)
    print(df)
    df = df[['accuracy_cv', 'accuracy_list', 'days_before_test', 'resolution', 'input', 'proba_y_false', 'proba_y_true', 'sliding_w', 'classifier', 'precision_true', 'precision_false']]
    df['a'] = df.apply(lambda row: sum([float(x) for x in str(row.precision_true).split(' ')]), axis=1)
    df = df.sort_values('a', ascending=False)
    print(df)
    cpt = 0
    for index, row in df.iterrows():
        proba_f, r_f = get_ranged_value('proba_y_false')
        proba_t, r_t = get_ranged_value('proba_y_true')
        accuracy, a_t = get_ranged_value('accuracy_list')
        prec_t, p_t = get_ranged_value('precision_true')
        prec_f, p_f = get_ranged_value('precision_false')
        input = row['input'].replace('-', '+').replace('humidity', 'h').replace('temperature', 't').replace('weight', 'w')
        line = "%.2f$\pm$%.2f&%.2f$\pm$%.2f&%.2f$\pm$%.2f&%.2f$\pm$%.2f&%.2f$\pm$%.2f&%d&%s&%s\\\\" % (accuracy, a_t,prec_t, p_t,prec_f, p_f, proba_f, r_f, proba_t, r_t, row['days_before_test'], row['resolution'], input)
        print(line)
        cpt += 1
        if cpt >= 59:
            cpt = 0
            print('\n\n\n')
    list_of_df = [g for _, g in df.groupby(['resolution', 'input', 'sliding_w', 'classifier'])]
    list_of_df_filtered = []
    for item in list_of_df:
        if item.shape[0] >= 6:
            item = item.reset_index(drop=True)
            item = item.sort_values('days_before_test', ascending=True)
            list_of_df_filtered.append(item)
            # print(item)
            fig = plt.figure()
            title = "%s_%s_%s" % (item['resolution'][0], item['input'][0], item['classifier'][0])
            plt.title(title)
            plt.plot(item['days_before_test'], item['accuracy_cv'])
            plt.show()
            pathlib.Path("result_analysis").mkdir(parents=True, exist_ok=True)
            path = "result_analysis\\" + title + ".png"
            fig.savefig(path)



