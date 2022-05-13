from pathlib import Path
FINAL_STR = ""


def main(
    log_dir="/user/work/fo18103/logs",
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
        ["QN"],
        ["QN", "STD"],
        ["QN", "ANSCOMBE", "LOG"],
        ["QN", "ANSCOMBE", "LOG", "CENTER", "CWTMORL"]
    ],
    meta_columns=[],
    individual_to_ignore=[],
):
    cpt = 0
    for steps in steps_list:
        cmd = f"ml.py --dataset-folder {dataset_folder} --n-imputed-days {i_day} --n-activity-days {a_day} --study-id {study_id} "

        for step in steps:
            cmd += f"--preprocessing-steps {step} "

        for meta in meta_columns:
            cmd += f"--meta-columns {meta} "

        for itoi in individual_to_ignore:
            cmd += f"--individual-to-ignore {itoi} "

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
        cmd += f"--output-dir {output_dir}/{study_id}_{cv}_{i_day}_{a_day}_{slug}_{clf_s}/{healthy_s}__{unhealthy_s}"
        # cmd += " > "+log_dir+"/${SLURM_ARRAY_TASK_ID}_"+f"{cv}_{i_day}_{a_day}_{slug}_{clf_s}__{healthy_s}__{unhealthy_s}"+".txt"

        print(cmd)
        global FINAL_STR
        FINAL_STR += f"'{cmd}' "
        cpt += 1
    return cpt


if __name__ == "__main__":
    n_cmd = 0
    output_dir = "/user/work/fo18103/cats_data/ml_build_multiple_peak_permutations"

    files = [x.stem for x in list(Path("E:/Cats/build_multiple_peak").glob("*"))]
    files = ["008__0_00100__120", "004__0_00100__120", "003__0_00100__120", "002__0_00100__120"]
    print(files)
    for t in files:
        for cv in ["LeaveOneOut"]:
            dataset_folder = f"/user/work/fo18103/cats_data/build_multiple_peak_permutations/{t}/dataset/training_sets/day_w"
            n_cmd += main(
                cv_list=[cv],
                output_dir=f"{output_dir}/{t}",
                dataset_folder=dataset_folder,
                a_day=-1,
                class_healthy_label_list=["0.0"],
                class_unhealthy_label_list=["1.0"],
                study_id="cats",
                meta_columns=[
                    "label",
                    "id",
                    "imputed_days",
                    "date",
                    "health",
                    "target",
                    "age",
                    "name",
                    "mobility_score",
                ],
                individual_to_ignore=["MrDudley", "Oliver_F", "Lucy"],
            )


    # cedara = "/user/work/fo18103/cedara/dataset6_mrnn_7day"
    # delmas = "/user/work/fo18103/delmas/dataset4_mrnn_7day"
    # output_dir = "/user/work/fo18103/thesis"
    # for activity_days in [1, 3, 7]:
    #     for imputed_days in [1, 3, 7]:
    #         n_cmd += main(
    #             output_dir=output_dir,
    #             dataset_folder=delmas,
    #             a_day=activity_days,
    #             i_day=imputed_days,
    #             class_healthy_label_list=["1To1"],
    #             class_unhealthy_label_list=["1To2"],
    #             study_id="delmas",
    #         )
    #         n_cmd += main(
    #             output_dir=output_dir,
    #             dataset_folder=delmas,
    #             a_day=activity_days,
    #             i_day=imputed_days,
    #             class_healthy_label_list=["1To1"],
    #             class_unhealthy_label_list=["2To2"],
    #             study_id="delmas",
    #         )
    #         n_cmd += main(
    #             output_dir=output_dir,
    #             dataset_folder=delmas,
    #             a_day=activity_days,
    #             i_day=imputed_days,
    #             class_healthy_label_list=["1To1"],
    #             class_unhealthy_label_list=["2To1"],
    #             study_id="delmas",
    #         )
    #         n_cmd += main(
    #             output_dir=output_dir,
    #             dataset_folder=delmas,
    #             a_day=activity_days,
    #             i_day=imputed_days,
    #             class_healthy_label_list=["1To1", "2To1"],
    #             class_unhealthy_label_list=["2To2", "1To2"],
    #             study_id="delmas",
    #         )
    #
    #         n_cmd += main(
    #             output_dir=output_dir,
    #             dataset_folder=cedara,
    #             a_day=activity_days,
    #             i_day=imputed_days,
    #             class_healthy_label_list=["1To1"],
    #             class_unhealthy_label_list=["1To2"],
    #             study_id="cedara",
    #         )
    #         n_cmd += main(
    #             output_dir=output_dir,
    #             dataset_folder=cedara,
    #             a_day=activity_days,
    #             i_day=imputed_days,
    #             class_healthy_label_list=["1To1"],
    #             class_unhealthy_label_list=["2To2"],
    #             study_id="cedara",
    #         )
    #         n_cmd += main(
    #             output_dir=output_dir,
    #             dataset_folder=cedara,
    #             a_day=activity_days,
    #             i_day=imputed_days,
    #             class_healthy_label_list=["1To1"],
    #             class_unhealthy_label_list=["2To1"],
    #             study_id="cedara",
    #         )
    #         n_cmd += main(
    #             output_dir=output_dir,
    #             dataset_folder=cedara,
    #             a_day=activity_days,
    #             i_day=imputed_days,
    #             class_healthy_label_list=["1To1", "2To1"],
    #             class_unhealthy_label_list=["2To2", "1To2"],
    #             study_id="cedara",
    #         )
    #         n_cmd += main(
    #             output_dir=output_dir,
    #             dataset_folder=cedara,
    #             a_day=activity_days,
    #             i_day=imputed_days,
    #             class_healthy_label_list=["1To1", "1To2", "2To1"],
    #             class_unhealthy_label_list=[
    #                 "4To4",
    #                 "3To5",
    #                 "4To3",
    #                 "5To3",
    #                 "2To5",
    #                 "2To2",
    #             ],
    #             study_id="cedara",
    #         )
            # n_cmd += main(
            #     output_dir=output_dir,
            #     dataset_folder=cedara,
            #     a_day=activity_days,
            #     class_healthy_label_list=["1To1"],
            #     class_unhealthy_label_list=[
            #         "4To4",
            #         "3To5",
            #         "4To3",
            #         "5To3",
            #         "2To5",
            #         "2To2",
            #     ],
            #     study_id="cedara",
            # )
    print(f"total cmd number is {n_cmd}")
    print(FINAL_STR)
