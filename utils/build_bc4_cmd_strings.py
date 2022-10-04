from pathlib import Path
import numpy as np
from sys import exit
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
        ["QN", "ANSCOMBE", "LOG"],
        ["QN", "ANSCOMBE", "LOG", "CENTER", "DWT"],
    ],
    individual_to_test=[603, 627, 77, 651, 607, 661, 621, 609, 622, 658],
    meta_columns=[],
    individual_to_ignore=[],
    n_scales= 9,
    sub_sample_scales= 3,
    batch_size = 10,
    epoch = 100,
    enable_qn_peak_filter = True
):
    cpt = 0
    for steps in steps_list:
        cmd = f"ml.py --batch-size {batch_size} --epoch {epoch} --no-pre-visu --dataset-folder {dataset_folder} --n-imputed-days {i_day} --n-activity-days {a_day} --study-id {study_id} --n-scales {n_scales} --sub-sample-scales {sub_sample_scales} "

        if enable_qn_peak_filter:
            cmd += "--enable-qn-peak-filter "

        for step in steps:
            cmd += f"--preprocessing-steps {step} "

        for itotest in individual_to_test:
            cmd += f"--individual-to-test {itotest} "

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


IDS = [
    "Greg",
    "Henry",
    "Tilly",
    "Maisie",
    "Sookie",
    "Oliver_F",
    "Ra",
    "Hector",
    "Jimmy",
    "MrDudley",
    "Kira",
    "Lucy",
    "Louis",
    "Luna_M",
    "Wookey",
    "Logan",
    "Ruby",
    "Kobe",
    "Saffy_J",
    "Enzo",
    "Milo",
    "Luna_F",
    "Oscar",
    "Kia",
    "Cat",
    "AlfieTickles",
    "Phoebe",
    "Harvey",
    "Mia",
    "Amadeus",
    "Marley",
    "Loulou",
    "Bumble",
    "Skittle",
    "Charlie_O",
    "Ginger",
    "Hugo_M",
    "Flip",
    "Guinness",
    "Chloe",
    "Bobby",
    "QueenPurr",
    "Jinx",
    "Charlie_B",
    "Thomas",
    "Sam",
    "Max",
    "Oliver_S",
    "Millie",
    "Clover",
    "Bobbie",
    "Gregory",
    "Kiki",
    "Hugo_R",
    "Shadow",
]


def cats():
    n_cmd = 0
    output_dir = "/user/work/fo18103/cats_data/ml_build_permutations_qnf_final"

    files = [x.stem for x in list(Path("E:/Cats/build_permutations").glob("*"))]
    files = [
        "1000__005__0_00100__030",
        "1000__005__0_00100__060",
        "1000__005__0_00100__120",
        "1000__006__0_00100__120",
        "2000__005__0_00100__030",
        "2000__005__0_00100__060",
        "2000__005__0_00100__120",
        "2000__006__0_00100__120",
        "3000__005__0_00100__030",
        "3000__005__0_00100__060",
        "3000__005__0_00100__120",
        "4000__005__0_00100__120",
        "5000__006__0_00100__120",
        "6000__005__0_00100__120",
        "7000__005__0_00100__120",
        "8000__005__0_00100__120",
        "800__006__0_00100__120",
        "9000__005__0_00100__120",
    ]
    #files = ["500__006__0_00100__120", "400__006__0_00100__120", "300__006__0_00100__120", "200__006__0_00100__120"]
    # files = [
    #     "1000__001__0_00100__060",
    #     "1000__002__0_00100__060",
    #     "1000__003__0_00100__060",
    #     "1000__004__0_00100__060",
    #     "1000__006__0_00100__060"
    # ]
    print(files)
    for t in files:
        for cv in ["LeaveOneOut"]:
            for cl in ["linear", "rbf"]:
                dataset_folder = f"/user/work/fo18103/cats_data/build_permutations/{t}/dataset/training_sets/samples"
                # for j in range(0, len(IDS), 5):
                #     print(np.unique(IDS[:j]+["MrDudley", "Oliver_F", "Lucy"]))
                for qn_filter in [True, False]:
                    n_cmd += main(
                        cv_list=[cv],
                        output_dir=f"{output_dir}/{t}",
                        dataset_folder=dataset_folder,
                        a_day=-1,
                        class_healthy_label_list=["0.0"],
                        class_unhealthy_label_list=["1.0"],
                        classifiers_list=[cl],
                        study_id="cats",
                        batch_size=20,
                        epoch=100,
                        #individual_to_test=[603, 627, 77, 651, 607, 661, 621, 609, 622, 658],
                        individual_to_test=[],
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
                        steps_list=[
                            ["QN"],
                            ["STD"],
                            ["QN", "STD"]
                        ],
                        individual_to_ignore=["MrDudley", "Oliver_F", "Lucy"],
                        enable_qn_peak_filter=qn_filter
                    )
    print(f"total cmd number is {n_cmd}")
    print(FINAL_STR)
    exit()


def cwt_sheep():
    n_cmd = 0
    cedara = "/user/work/fo18103/cedara/dataset6_mrnn_7day_clipped"
    delmas = "/user/work/fo18103/delmas/dataset4_mrnn_7day"
    output_dir = "/user/work/fo18103/thesis/cwt_optimal_exp"

    for cl in ["linear", "rbf"]:
        for scales in [6, 9, 12, 18]:
            for sub in [1, 3, 6]:
                n_cmd += main(
                    output_dir=output_dir,
                    dataset_folder=delmas,
                    a_day=3,
                    i_day=3,
                    class_healthy_label_list=["1To1"],
                    class_unhealthy_label_list=["2To2"],
                    study_id="delmas",
                    steps_list=[["QN", "ANSCOMBE", "LOG", "CENTER", f"CWTMORLs{scales}s{sub}"]],
                    n_scales=scales,
                    classifiers_list=[cl],
                    sub_sample_scales=sub
                )
        n_cmd += main(
            output_dir=output_dir,
            dataset_folder=delmas,
            a_day=3,
            i_day=3,
            class_healthy_label_list=["1To1"],
            class_unhealthy_label_list=["2To2"],
            study_id="delmas",
            classifiers_list=[cl],
            steps_list=[["QN", "ANSCOMBE", "LOG", "CENTER", "DWT"]]
        )
        n_cmd += main(
            output_dir=output_dir,
            dataset_folder=delmas,
            a_day=3,
            i_day=3,
            class_healthy_label_list=["1To1"],
            class_unhealthy_label_list=["2To2"],
            study_id="delmas",
            classifiers_list=[cl],
            steps_list=[["QN", "ANSCOMBE", "LOG"]]
        )
    print(f"total cmd number is {n_cmd}")
    print(FINAL_STR)


def goat_sheep():
    n_cmd = 0
    cedara = "/user/work/fo18103/cedara/dataset6_mrnn_7day_clipped"
    delmas = "/user/work/fo18103/delmas/dataset4_mrnn_7day"
    output_dir = "/user/work/fo18103/thesis"
    for activity_days in [1, 3, 7]:
        for imputed_days in [1, 3, 7]:
            n_cmd += main(
                output_dir=output_dir,
                dataset_folder=cedara,
                a_day=activity_days,
                i_day=imputed_days,
                class_healthy_label_list=["1To1"],
                class_unhealthy_label_list=["1To2"],
                study_id="cedara",
            )
            n_cmd += main(
                output_dir=output_dir,
                dataset_folder=cedara,
                a_day=activity_days,
                i_day=imputed_days,
                class_healthy_label_list=["1To1"],
                class_unhealthy_label_list=["2To2"],
                study_id="cedara",
            )
            n_cmd += main(
                output_dir=output_dir,
                dataset_folder=cedara,
                a_day=activity_days,
                i_day=imputed_days,
                class_healthy_label_list=["1To1"],
                class_unhealthy_label_list=["2To1"],
                study_id="cedara",
            )
            n_cmd += main(
                output_dir=output_dir,
                dataset_folder=cedara,
                a_day=activity_days,
                i_day=imputed_days,
                class_healthy_label_list=["1To1", "2To1"],
                class_unhealthy_label_list=["2To2", "1To2"],
                study_id="cedara",
            )
            n_cmd += main(
                output_dir=output_dir,
                dataset_folder=cedara,
                a_day=activity_days,
                i_day=imputed_days,
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
                output_dir=output_dir,
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

            n_cmd += main(
                output_dir=output_dir,
                dataset_folder=delmas,
                a_day=activity_days,
                i_day=imputed_days,
                class_healthy_label_list=["1To1"],
                class_unhealthy_label_list=["1To2"],
                study_id="delmas",
            )
            n_cmd += main(
                output_dir=output_dir,
                dataset_folder=delmas,
                a_day=activity_days,
                i_day=imputed_days,
                class_healthy_label_list=["1To1"],
                class_unhealthy_label_list=["2To2"],
                study_id="delmas",
            )
            n_cmd += main(
                output_dir=output_dir,
                dataset_folder=delmas,
                a_day=activity_days,
                i_day=imputed_days,
                class_healthy_label_list=["1To1"],
                class_unhealthy_label_list=["2To1"],
                study_id="delmas",
            )
            n_cmd += main(
                output_dir=output_dir,
                dataset_folder=delmas,
                a_day=activity_days,
                i_day=imputed_days,
                class_healthy_label_list=["1To1", "2To1"],
                class_unhealthy_label_list=["2To2", "1To2"],
                study_id="delmas",
            )

    print(f"total cmd number is {n_cmd}")
    print(FINAL_STR)


if __name__ == "__main__":
    #cwt_sheep()
    cats()
    # goat_sheep()