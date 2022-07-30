import typer
import ml as main_experiment
import ml_cross_farm_validation as cross_farm_validation
import ml_temporal_validation as temporal_validation
from pathlib import Path


def local_run():
    main(output_dir=Path("E:/thesis_debug_mrnn9/"), delmas_dir=Path("E:/thesis/datasets/delmas/datasetmrnn7_9"))
    # main(output_dir=Path("E:/thesis_debug_mrnn18/"), delmas_dir=Path("E:/thesis/datasets/delmas/datasetmrnn7_18"))
    # main(output_dir=Path("E:/thesis_debug_mrnn19/"), delmas_dir=Path("E:/thesis/datasets/delmas/datasetmrnn7_19"))
    # main(output_dir=Path("E:/thesis_debug_mrnn20/"), delmas_dir=Path("E:/thesis/datasets/delmas/datasetmrnn7_20"))
    # main(output_dir=Path("E:/thesis_debug_mrnn21/"), delmas_dir=Path("E:/thesis/datasets/delmas/datasetmrnn7_21"))
    # main(output_dir=Path("E:/thesis_debug_mrnn22/"), delmas_dir=Path("E:/thesis/datasets/delmas/datasetmrnn7_22"))


def main(
    exp_main: bool = True,
    exp_temporal: bool = False,
    exp_cross_farm: bool = False,
    output_dir: Path = Path("E:/thesis_debug_mrnn19/"),
    delmas_dir: Path = Path("E:/thesis/datasets/delmas/datasetmrnn7_19"),
    cedara_dir: Path = Path("E:/Data2/debug3/cedara/dataset6_mrnn_7day"),
):
    """Thesis script runs all key experiments for data exploration chapter
    Args:\n
        output_dir: Output directory
    """

    if exp_main:
        print("experiment 1: main pipeline")

        steps_list = [
            # ["LINEAR", "QN", "STD"],
            # ["LINEAR", "QN", "ANSCOMBE", "STD"],
            # ["LINEAR", "QN", "LOG", "STD"],
            ["QN", "ANSCOMBE", "LOG"]
            #["QN", "ANSCOMBE", "LOG", "CENTER", "DWT"]
            # ["QN", "ANSCOMBE", "LOG", "STD", "APPEND", "LINEAR", "QN", "ANSCOMBE", "LOG", "CENTER", "DWT"],
            #["QN", "ANSCOMBE", "LOG", "CENTER", "DWT"],
            # ["QN", "ANSCOMBE", "LOG", "STD", "APPEND", "LINEAR", "QN", "ANSCOMBE", "LOG", "CENTER", "CWTMORL"],
            #["QN", "ANSCOMBE", "LOG", "CENTER", "CWTMORL"],
            # ["LINEAR", "QN", "LOG", "CENTER", "CWT(MORL)"],
            # ["LINEAR", "QN", "ANSCOMBE", "LOG", "CENTER", "CWT(MORL)", "STD"]
        ]
        for steps in steps_list:
            slug = "_".join(steps)

            for i_day in [1, 7]:
                for a_day in [1, 2, 3, 4, 5, 6, 7]:
                    for cv in ['RepeatedKFold']:
                        for add_seasons_to_features in [False]:

                            # main_experiment.main(
                            #     output_dir=output_dir
                            #     / "main_experiment_"
                            #     / f"delmas_{cv}_{i_day}_{a_day}_{slug}_season_{add_seasons_to_features}"
                            #     / "2To2",
                            #     dataset_folder=delmas_dir,
                            #     preprocessing_steps=steps,
                            #     n_imputed_days=i_day,
                            #     n_activity_days=a_day,
                            #     cv=cv,
                            #     classifiers=["linear", "rbf"],
                            #     class_unhealthy_label=["2To2"],
                            #     study_id="delmas",
                            #     add_seasons_to_features=add_seasons_to_features,
                            #     sampling="T",
                            #     resolution=-1
                            # )

                            main_experiment.main(
                                output_dir=output_dir
                                / "main_experiment"
                                / f"delmas_{cv}_{i_day}_{a_day}_{slug}_season_{add_seasons_to_features}"
                                / "2To2",
                                dataset_folder=delmas_dir,
                                preprocessing_steps=steps,
                                n_imputed_days=i_day,
                                n_activity_days=a_day,
                                cv=cv,
                                classifiers=["linear", "rbf"],
                                class_unhealthy_label=["2To2"],
                                study_id="delmas",
                                add_seasons_to_features=add_seasons_to_features
                            )

                            # main_experiment.main(
                            #     output_dir=output_dir
                            #     / "main_experiment"
                            #     / f"delmas_{cv}_{i_day}_{a_day}_{slug}_season_{add_seasons_to_features}"
                            #     / "1To1_2To1__2To2",
                            #     dataset_folder=delmas_dir,
                            #     preprocessing_steps=steps,
                            #     n_imputed_days=i_day,
                            #     n_activity_days=a_day,
                            #     cv=cv,
                            #     classifiers=["linear", "rbf"],
                            #     class_healthy_label=["1To1", "2To1"],
                            #     class_unhealthy_label=["2To2"],
                            #     study_id="delmas",
                            #     add_seasons_to_features=add_seasons_to_features,
                            # )
                            #
                            # main_experiment.main(
                            #     output_dir=output_dir
                            #     / "main_experiment"
                            #     / f"delmas_{cv}_{i_day}_{a_day}_{slug}_season_{add_seasons_to_features}"
                            #     / "1To1_2To1__2To2_1To2",
                            #     dataset_folder=delmas_dir,
                            #     preprocessing_steps=steps,
                            #     n_imputed_days=i_day,
                            #     n_activity_days=a_day,
                            #     cv=cv,
                            #     classifiers=["linear", "rbf"],
                            #     class_healthy_label=["1To1", "2To1"],
                            #     class_unhealthy_label=["2To2", "1To2"],
                            #     study_id="delmas",
                            #     add_seasons_to_features=add_seasons_to_features,
                            # )
                            #
                            main_experiment.main(
                                output_dir=output_dir
                                / "main_experiment"
                                / f"delmas_{cv}_{i_day}_{a_day}_{slug}_season_{add_seasons_to_features}"
                                / "1To2",
                                dataset_folder=delmas_dir,
                                preprocessing_steps=steps,
                                n_imputed_days=i_day,
                                n_activity_days=a_day,
                                class_unhealthy_label=["1To2"],
                                cv=cv,
                                classifiers=["linear", "rbf"],
                                study_id="delmas",
                                add_seasons_to_features=add_seasons_to_features
                            )

                            # main_experiment.main(
                            #     output_dir=output_dir
                            #     / "main_experiment"
                            #     / f"delmas_{cv}_{i_day}_{a_day}_{slug}_season_{add_seasons_to_features}"
                            #     / "2To1",
                            #     dataset_folder=delmas_dir,
                            #     preprocessing_steps=steps,
                            #     n_imputed_days=i_day,
                            #     n_activity_days=a_day,
                            #     class_unhealthy_label=["2To1"],
                            #     cv=cv,
                            #     classifiers=["linear", "rbf"],
                            #     study_id="delmas",
                            #     add_seasons_to_features=add_seasons_to_features
                            # )
                            #
                            # main_experiment.main(
                            #     output_dir=output_dir
                            #     / "main_experiment"
                            #     / f"cedara_{cv}_{i_day}_{a_day}_{slug}_season_{add_seasons_to_features}"
                            #     / "2To2",
                            #     dataset_folder=cedara_dir,
                            #     preprocessing_steps=steps,
                            #     n_imputed_days=i_day,
                            #     n_activity_days=a_day,
                            #     class_unhealthy_label=["2To2"],
                            #     cv=cv,
                            #     classifiers=["linear", "rbf"],
                            #     study_id="cedara",
                            #     add_seasons_to_features=add_seasons_to_features
                            # )
                            #
                            # main_experiment.main(
                            #     output_dir=output_dir
                            #     / "main_experiment"
                            #     / f"cedara_{cv}_{i_day}_{a_day}_{slug}_season_{add_seasons_to_features}"
                            #     / "1To1_1To2_2To1__2To2",
                            #     dataset_folder=cedara_dir,
                            #     preprocessing_steps=steps,
                            #     n_imputed_days=i_day,
                            #     n_activity_days=a_day,
                            #     class_healthy_label=["1To1", "1To2", "2To1"],
                            #     class_unhealthy_label=["2To2"],
                            #     cv=cv,
                            #     classifiers=["linear", "rbf"],
                            #     study_id="cedara",
                            #     add_seasons_to_features=add_seasons_to_features
                            # )
                            #
                            # main_experiment.main(
                            #     output_dir=output_dir
                            #     / "main_experiment"
                            #     / f"cedara_{cv}_{i_day}_{a_day}_{slug}_season_{add_seasons_to_features}"
                            #     / "1To1_1To2_2To1__4To4_3To5_4To3_5To3_2To5_2To2",
                            #     dataset_folder=cedara_dir,
                            #     preprocessing_steps=steps,
                            #     n_imputed_days=i_day,
                            #     n_activity_days=a_day,
                            #     class_healthy_label=["1To1", "1To2", "2To1"],
                            #     class_unhealthy_label=[
                            #         "4To4",
                            #         "3To5",
                            #         "4To3",
                            #         "5To3",
                            #         "2To5",
                            #         "2To2",
                            #     ],
                            #     cv=cv,
                            #     classifiers=["linear", "rbf"],
                            #     study_id="cedara",
                            #     add_seasons_to_features=add_seasons_to_features
                            # )
                            #
                            # main_experiment.main(
                            #     output_dir=output_dir
                            #     / "main_experiment"
                            #     / f"cedara_{cv}_{i_day}_{a_day}_{slug}_season_{add_seasons_to_features}"
                            #     / "4To4_3To5_4To3_5To3_2To5_2To2",
                            #     dataset_folder=cedara_dir,
                            #     preprocessing_steps=steps,
                            #     n_imputed_days=i_day,
                            #     n_activity_days=a_day,
                            #     class_unhealthy_label=[
                            #         "4To4",
                            #         "3To5",
                            #         "4To3",
                            #         "5To3",
                            #         "2To5",
                            #         "2To2",
                            #     ],
                            #     cv=cv,
                            #     classifiers=["linear", "rbf"],
                            #     study_id="cedara",
                            #     add_seasons_to_features=add_seasons_to_features
                            # )
                            #
                            # main_experiment.main(
                            #     output_dir=output_dir
                            #     / "main_experiment"
                            #     / f"cedara_{cv}_{i_day}_{a_day}_{slug}_season_{add_seasons_to_features}"
                            #     / "1To2",
                            #     dataset_folder=cedara_dir,
                            #     preprocessing_steps=steps,
                            #     n_imputed_days=i_day,
                            #     n_activity_days=a_day,
                            #     class_unhealthy_label=["1To2"],
                            #     cv=cv,
                            #     classifiers=["linear", "rbf"],
                            #     study_id="cedara",
                            #     add_seasons_to_features=add_seasons_to_features
                            # )
                            #
                            # main_experiment.main(
                            #     output_dir=output_dir
                            #     / "main_experiment"
                            #     / f"cedara_{cv}_{i_day}_{a_day}_{slug}_season_{add_seasons_to_features}"
                            #     / "2To1_3To1",
                            #     dataset_folder=cedara_dir,
                            #     preprocessing_steps=steps,
                            #     n_imputed_days=i_day,
                            #     n_activity_days=a_day,
                            #     class_unhealthy_label=["2To1", "3To1"],
                            #     cv=cv,
                            #     classifiers=["linear", "rbf"],
                            #     study_id="cedara",
                            #     add_seasons_to_features=add_seasons_to_features
                            # )

    if exp_temporal:
        print("experiment 2: temporal validation")
        for i in [7, 6]:
            for n_a in [1, 3, 7]:
                temporal_validation.main(
                    output_dir=output_dir / "temporal_validation" / f"delmas_{i}_{n_a}" / "2To2",
                    dataset_folder=delmas_dir,
                    n_imputed_days=i,
                    n_activity_days=n_a
                )

                temporal_validation.main(
                    output_dir=output_dir / "temporal_validation" / f"cedara_{i}_{n_a}" / "2To2",
                    dataset_folder=cedara_dir,
                    n_imputed_days=i,
                    n_activity_days=n_a
                )

    if exp_cross_farm:
        print("experiment 3: cross farm validation")
        for imp_d in [7, 6]:
            for a_act_day in [7]:
                cross_farm_validation.main(
                    farm1_path=delmas_dir,
                    farm2_path=cedara_dir,
                    output_dir=output_dir / "cross_farm" / f"{imp_d}_{a_act_day}" / "2To2",
                    n_imputed_days=imp_d,
                    n_activity_days=a_act_day,
                    class_unhealthy_f2=["2To2"],
                )

                # cross_farm_validation.main(
                #     farm1_path=Path("E:\Data2\debug3\delmas\dataset4_mrnn_7day"),
                #     farm2_path=Path("E:\Data2\debug3\cedara\dataset6_mrnn_7day"),
                #     output_dir=output_dir
                #     / "cross_Farm"
                #     / f"{imp_d}_{a_act_day}"
                #     / "4To4_3To5_4To3_5To3_2To5_2To2",
                #     n_imputed_days=imp_d,
                #     n_activity_days=a_act_day,
                #     class_unhealthy_f2=[
                #         "4To4",
                #         "3To5",
                #         "4To3",
                #         "5To3",
                #         "2To5",
                #         "2To2",
                #     ],
                # )


if __name__ == "__main__":
    local_run()
    #typer.run(main)
