import json
import statistics


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


if __name__ == '__main__':
    cedara_weigh_list = [43.9, 45.0, 46.7,50.2,50.4,48.4,46.6,46.3,43.4,40.4,40.5,40.3,39.0,41.9,39.4,40.2,40.5,40.6,40.7,41.9,41.7,44.5,44.3,43.9,43.9,44.0,45.8,46.9,48.0,46.9,48.1,48.2,49.8,49.2,50.7,42.9,43.3,42.0,40.7,41.1,41.9,43.1,43.2]
    mean_weight = statistics.mean(cedara_weigh_list)
    print("cedara mean weight=%f" % mean_weight)

    print("start")
    for file in ['delmas_famacha_data.json', 'cedara_famacha_data.json']:
        with open(file) as fp:
            data_famacha_dict = json.load(fp)
            cpt_change = 0
            cpt_no_change = 0
            weights = []
            for key, value in data_famacha_dict.items():
                for i in range(len(value) - 1):
                    curr = value[i]
                    next = value[i+1]

                    if is_number(curr[3]):
                        weights.append(curr[3])


                    date1 = curr[0].split('/')
                    date2 = next[0].split('/')

                    if 'delmas' in file:
                        if int(date2[1]) > 4 and int(date2[2]) == 2016:
                            continue
                        if int(date1[1]) < 3 and int(date2[2]) == 2015:
                            continue

                    if 'cedara' in file:
                        if int(date2[1]) > 7 and int(date2[2]) == 2013:
                            continue
                        if int(date1[1]) < 4 and int(date2[2]) == 2012:
                            continue


                    famacha1 = curr[1]
                    famacha2 = next[1]
                    try:
                        int(famacha2)
                        int(famacha1)
                    except ValueError as e:
                        continue
                    if (famacha2 == famacha1):
                        cpt_change = cpt_change + 1
                    else:
                        cpt_no_change = cpt_no_change + 1
            mean_weight = statistics.mean(weights)
            print("mean weight=%f" % mean_weight)
            print("cpt_change=%d cpt_no_change=%d" % (cpt_change, cpt_no_change))

