def main(
    output_dir="",
    dataset_folder="",
    i_day=-1,
    a_day=-1,
    cv_list=["RepeatedKFold"],
    classifiers_list=["linear", "rbf"],
    class_healthy_label_list=["1To1"],
    class_unhealthy_label_list=["2To2"],
    study_id="delmas",
    steps_list=[
        ["LINEAR"],
        ["LINEAR", "QN"],
        ["LINEAR", "QN", "ANSCOMBE"],
        ["LINEAR", "QN", "LOG"],
        ["LINEAR", "QN", "ANSCOMBE", "LOG"],
        ["LINEAR", "QN", "ANSCOMBE", "LOG", "CENTER", "CWT(MORL)"],
        ["LINEAR", "QN", "ANSCOMBE", "CENTER", "CWT(MORL)"],
        ["LINEAR", "QN", "LOG", "CENTER", "CWT(MORL)"],
    ],
):
    cpt = 0
    for steps in steps_list:
        cmd = f"python ml.py --dataset-folder {dataset_folder} --n-imputed-days {i_day} --n-activity-days {a_day} --study-id {study_id} "

        for step in steps:
            cmd += f"--preprocessing-steps {step} "

        for cv in cv_list:
            cmd += f"--cv {cv} "

        for clf in classifiers_list:
            cmd += f"--classifiers {clf} "

        for class_healthy_label in class_healthy_label_list:
            cmd += f"--class-healthy-label {class_healthy_label} "

        for class_unhealthy_label in class_unhealthy_label_list:
            cmd += f"--class-unhealthy-label {class_unhealthy_label} "

        slug = "_".join(steps)
        healthy_s = "_".join(class_healthy_label_list)
        unhealthy_s = "_".join(class_unhealthy_label_list)
        clf_s = "_".join(classifiers_list)
        cmd += f"--output-dir {output_dir}/main_experiment/delmas_{cv}_{i_day}_{a_day}_{slug}_{clf_s}/{healthy_s}__{unhealthy_s}"
        cmd += " > ${SLURM_ARRAY_TASK_ID}_"+f"{cv}_{i_day}_{a_day}_{slug}_{clf_s}__{healthy_s}__{unhealthy_s}"+".txt"

        print(cmd)
        cpt += 1
    return cpt


if __name__ == "__main__":
    n_cmd = 0
    cedara = "/user/work/fo18103/cedara/datasetraw_none_7day"
    delmas = "/user/work/fo18103/delmas/datasetraw_none_7day"
    for activity_days in [1, 2, 3, 4, 5, 6, 7]:
        n_cmd += main(
            dataset_folder=delmas,
            a_day=activity_days,
            class_healthy_label_list=["1To1"],
            class_unhealthy_label_list=["1To2"],
            study_id="delmas",
        )
        n_cmd += main(
            dataset_folder=delmas,
            a_day=activity_days,
            class_healthy_label_list=["1To1"],
            class_unhealthy_label_list=["2To2"],
            study_id="delmas",
        )
        n_cmd += main(
            dataset_folder=delmas,
            a_day=activity_days,
            class_healthy_label_list=["1To1"],
            class_unhealthy_label_list=["2To1"],
            study_id="delmas",
        )
        n_cmd += main(
            dataset_folder=delmas,
            a_day=activity_days,
            class_healthy_label_list=["1To1", "2To1"],
            class_unhealthy_label_list=["2To2", "1To2"],
            study_id="delmas",
        )

        n_cmd += main(
            dataset_folder=cedara,
            a_day=activity_days,
            class_healthy_label_list=["1To1"],
            class_unhealthy_label_list=["1To2"],
            study_id="cedara",
        )
        n_cmd += main(
            dataset_folder=cedara,
            a_day=activity_days,
            class_healthy_label_list=["1To1"],
            class_unhealthy_label_list=["2To2"],
            study_id="cedara",
        )
        n_cmd += main(
            dataset_folder=cedara,
            a_day=activity_days,
            class_healthy_label_list=["1To1"],
            class_unhealthy_label_list=["2To1"],
            study_id="cedara",
        )
        n_cmd += main(
            dataset_folder=cedara,
            a_day=activity_days,
            class_healthy_label_list=["1To1", "2To1"],
            class_unhealthy_label_list=["2To2", "1To2"],
            study_id="cedara",
        )
        n_cmd += main(
            dataset_folder=cedara,
            a_day=activity_days,
            class_healthy_label_list=["1To1", "1To2", "2To1"],
            class_unhealthy_label_list=["2To2"],
            study_id="cedara",
        )
        n_cmd += main(
            dataset_folder=cedara,
            a_day=activity_days,
            class_healthy_label_list=["1To1", "1To2", "2To1"],
            class_unhealthy_label_list=[
                "4To4",
                "3To5",
                "4To3",
                "5To3",
                "2To5",
                "2To2",
            ],
            study_id="cedara",
        )
        n_cmd += main(
            dataset_folder=cedara,
            a_day=activity_days,
            class_healthy_label_list=["1To1"],
            class_unhealthy_label_list=[
                "4To4",
                "3To5",
                "4To3",
                "5To3",
                "2To5",
                "2To2",
            ],
            study_id="cedara",
        )
    print(f"total cmd number is {n_cmd}")
