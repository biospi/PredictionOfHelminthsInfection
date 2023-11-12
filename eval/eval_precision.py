import json
import numpy as np

def main(path):
    with open(path) as f:
        d = json.load(f)
        data = d["SVC_linear_results"]
        tps0_list = []
        tps1_list = []
        for item in data:
            tps0 = item["test_precision_score_0"]
            tps1 = item["test_precision_score_1"]
            tps0_list.append(tps0)
            tps1_list.append(tps1)
        print(f"tps0= {np.median(tps0_list)}, tps1 = {np.median(tps1_list)}")


if __name__ == "__main__":
    main("E:/thesis_ltwoo/main_experiment/linear/delmas_dataset4_mrnn_7day_delmas_LeaveTwoOut_6_7_7_QN_ANSCOMBE_LOG_season_False_1.0_scale/2To2/results.json")