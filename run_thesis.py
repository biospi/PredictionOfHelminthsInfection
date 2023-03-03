import os

import typer
import ml as main_experiment
import ml_cross_farm_validation as cross_farm_validation
import ml_temporal_validation as temporal_validation
from pathlib import Path


def single_run(
    output_dir=Path("E:/thesis_final_march1"),
    clf="rbf",
    farm_id="",
    cv="RepeatedKFold",
    i_day=1,
    a_day=1,
    w_day=7,
    add_seasons_to_features=False,
    class_unhealthy_label="2To2",
    n_job=6,
    dataset=None,
    export_hpc_string=False
):
    steps = ["QN", "ANSCOMBE", "LOG"]
    slug = "_".join(steps)
    main_experiment.main(
        output_dir=output_dir
        / "main_experiment"
        / clf
        / f"{dataset.stem}_{farm_id}_{cv}_{i_day}_{a_day}_{w_day}_{slug}_season_{add_seasons_to_features}"
        / class_unhealthy_label,
        dataset_folder=dataset,
        preprocessing_steps=steps,
        n_imputed_days=i_day,
        n_activity_days=a_day,
        n_weather_days=w_day,
        cv=cv,
        classifiers=[clf],
        class_unhealthy_label=[class_unhealthy_label],
        study_id=farm_id,
        add_seasons_to_features=add_seasons_to_features,
        export_fig_as_pdf=False,
        plot_2d_space=False,
        pre_visu=True,
        skip=False,
        export_hpc_string=export_hpc_string,
        weather_file=Path(
            "C:/Users/fo18103/PycharmProjects/PredictionOfHelminthsInfection/weather_data/delmas_south_africa_2011-01-01_to_2015-12-31.csv"
        ),
        n_job=n_job,
    )


def biospi_run(n_job=25):
    main(
        output_dir=Path("/mnt/storage/scratch/axel/thesis"),
        cedara_dir_mrnn=Path(
            "/mnt/storage/scratch/axel/thesis/datasets/cedara/cedara_dataset6_mrnn_7day"
        ),
        # cedara_dir_gain=Path(
        #     "/mnt/storage/scratch/axel/thesis/datasets/cedara/cedara_dataset_1_gain_60"
        # ),
        # cedara_dir_li=Path(
        #     "/mnt/storage/scratch/axel/thesis/datasets/cedara/cedara_dataset_1_li_60"
        # ),
        delmas_dir_mrnn=Path(
            "/mnt/storage/scratch/axel/thesis/datasets/delmas/delmas_dataset4_mrnn_7day"
        ),
        # delmas_dir_gain=Path(
        #     "/mnt/storage/scratch/axel/thesis/datasets/delmas/delmas_dataset_1_gain_60"
        # ),
        # delmas_dir_li=Path(
        #     "/mnt/storage/scratch/axel/thesis/datasets/delmas/delmas_dataset_1_li_60"
        # ),
        n_job=n_job,
    )


def local_run():

    main(
        output_dir=Path("E:/thesis_final_march1"),
        cedara_dir_mrnn=Path("E:/thesis/datasets/cedara/cedara_datasetmrnn7_23"),
        cedara_dir_gain=Path("E:/thesis/datasets/cedara/cedara_dataset_1_gain_172_no_filter_fixed"),
        cedara_dir_li=Path("E:/thesis/datasets/cedara/cedara_dataset_1_li_172_no_filter_fixed"),
        delmas_dir_mrnn=Path("E:/thesis/datasets/delmas/delmas_dataset4_mrnn_7day"),
        delmas_dir_gain=Path("E:/thesis/datasets/delmas/delmas_dataset_1_gain_66_no_filter_fixed"),
        delmas_dir_li=Path("E:/thesis/datasets/delmas/delmas_dataset_1_li_66_no_filter_fixed"),
        export_hpc_string=True
    )
    # main(output_dir=Path("E:/thesis_debug_mrnn18/"), delmas_dir=Path("E:/thesis/datasets/delmas/datasetmrnn7_18"))
    # main(output_dir=Path("E:/thesis_debug_mrnn19/"), delmas_dir=Path("E:/thesis/datasets/delmas/datasetmrnn7_19"))
    # main(output_dir=Path("E:/thesis_debug_mrnn20/"), delmas_dir=Path("E:/thesis/datasets/delmas/datasetmrnn7_20"))
    # main(output_dir=Path("E:/thesis_debug_mrnn21/"), delmas_dir=Path("E:/thesis/datasets/delmas/datasetmrnn7_21"))
    # main(output_dir=Path("E:/thesis_debug_mrnn22/"), delmas_dir=Path("E:/thesis/datasets/delmas/datasetmrnn7_22"))


def main(
    exp_main: bool = True,
    exp_temporal: bool = False,
    exp_cross_farm: bool = False,
    weather_exp: bool = False,
    output_dir: Path = Path("E:/thesis"),
    delmas_dir_mrnn: Path = None,
    delmas_dir_gain: Path = None,
    delmas_dir_li: Path = None,
    cedara_dir_mrnn: Path = None,
    cedara_dir_gain: Path = None,
    cedara_dir_li: Path = None,
    n_job: int = 6,
    export_hpc_string: bool = False
):
    """Thesis script runs all key experiments for data exploration chapter
    Args:\n
        output_dir: Output directory
    """

    if weather_exp:
        print("experiment 1 (weather): main pipeline")

        steps_list = [
            ["WINDSPEED", "STDS"],
            ["HUMIDITY", "STDS"],
            ["RAINFALL", "STDS"],
            ["TEMPERATURE", "STDS"],
        ]
        for steps in steps_list:
            slug = "_".join(steps)
            for w_day in [7]:
                for cv in ["RepeatedKFold"]:
                    for add_seasons_to_features in [False]:
                        # main_experiment.main(
                        #     output_dir=output_dir
                        #     / "main_experiment"
                        #     / f"delmas_{cv}_{i_day}_{a_day}_{w_day}_{slug}_season_{add_seasons_to_features}"
                        #     / "2To2",
                        #     dataset_folder=delmas_dir,
                        #     preprocessing_steps=steps,
                        #     n_imputed_days=i_day,
                        #     n_activity_days=a_day,
                        #     n_weather_days=w_day,
                        #     cv=cv,
                        #     classifiers=["rbf", "linear"],
                        #     class_unhealthy_label=["2To2"],
                        #     study_id="delmas",
                        #     add_seasons_to_features=add_seasons_to_features,
                        #     export_fig_as_pdf=True,
                        #     plot_2d_space=True,
                        #     weather_file=Path(
                        #         "C:/Users/fo18103/PycharmProjects/PredictionOfHelminthsInfection/weather_data/delmas_south_africa_2011-01-01_to_2015-12-31.csv"),
                        # )
                        main_experiment.main(
                            output_dir=output_dir
                            / "main_experiment"
                            / f"cedara_{cv}_{0}_{0}_{w_day}_{slug}_season_{add_seasons_to_features}"
                            / "2To2",
                            dataset_folder=cedara_dir_mrnn,
                            preprocessing_steps=steps,
                            n_imputed_days=0,
                            n_activity_days=0,
                            n_weather_days=w_day,
                            class_unhealthy_label=["2To2"],
                            cv=cv,
                            classifiers=["linear", "rbf"],
                            study_id="cedara",
                            add_seasons_to_features=add_seasons_to_features,
                            plot_2d_space=False,
                            export_fig_as_pdf=False,
                            pre_visu=True,
                            export_hpc_string=export_hpc_string,
                            weather_file=Path(
                                "C:/Users/fo18103/PycharmProjects/PredictionOfHelminthsInfection/weather_data/cedara_south_africa_2011-01-01_to_2015-12-31.csv"
                            ),
                        )
                        continue

    if exp_main:
        print("experiment 1: main pipeline")

        steps_list = [
            [],
            ["QN"],
            ["QN", "ANSCOMBE", "LOG"],
            ["QN", "ANSCOMBE", "LOG", "STD"],
            ["QN", "ANSCOMBE", "LOG", "MINMAX"],
            ["QN", "ANSCOMBE", "LOG", "CENTER", "CWT", "STD"],

            # ["L2"],
            # ["L2", "ANSCOMBE", "LOG"],
            # ["L2", "ANSCOMBE", "LOG", "STD"],
            # ["L2", "ANSCOMBE", "LOG", "CENTER", "CWT", "STD"],
            # ["L2"],
            # ["QN", "ANSCOMBE", "LOG", "STD"],
            # ["QN", "ANSCOMBE", "LOG", "MINMAX"],
            # ["QN", "ANSCOMBE", "LOG", "CWT", "STD"],
            # ["LINEAR", "QN", "STD"],
            # ["LINEAR", "QN", "ANSCOMBE", "STD"],
            # ["LINEAR", "QN", "LOG", "STD"],
            # ["QN", "ANSCOMBE", "LOG", "HUMIDITYAPPEND", "TEMPERATUREAPPEND", "STDS"],
            # ["QN", "ANSCOMBE", "LOG", "HUMIDITYAPPEND", "STDS"],
            # ["QN", "ANSCOMBE", "LOG", "TEMPERATUREAPPEND", "STDS"],
            # ["QN", "ANSCOMBE", "LOG", "RAINFALLAPPEND", "STDS"],
            # ["QN", "ANSCOMBE", "LOG", "WINDSPEEDAPPEND", "STDS"],
            # ["QN", "ANSCOMBE", "LOG"]
            # ["QN", "ANSCOMBE", "LOG", "STDS"],
            # ["QN", "ANSCOMBE", "LOG", "CENTER", "DWT"]
            # ["QN", "ANSCOMBE", "LOG", "STD", "APPEND", "LINEAR", "QN", "ANSCOMBE", "LOG", "CENTER", "DWT"],
            # ["QN", "ANSCOMBE", "LOG", "CENTER", "DWT"],
            # ["QN", "ANSCOMBE", "LOG", "STD", "APPEND", "LINEAR", "QN", "ANSCOMBE", "LOG", "CENTER", "CWTMORL"],
            # ["QN", "ANSCOMBE", "LOG", "CENTER", "CWTMORL"],
            # ["LINEAR", "QN", "LOG", "CENTER", "CWT(MORL)"],
            # ["LINEAR", "QN", "ANSCOMBE", "LOG", "CENTER", "CWT(MORL)", "STD"]
        ]
        for class_unhealthy_label in ["2To2"]:
            for steps in steps_list:
                slug = "_".join(steps)
                for clf in ["linear", "rbf", "knn", "lreg", "dtree"]:
                    for i_day in [1, 4, 7]:
                        for a_day in [1, 4, 7]:
                            for w_day in [7]:
                                for cv in ["RepeatedKFold"]:
                                    for add_seasons_to_features in [False]:
                                        for dataset in [
                                            delmas_dir_mrnn,
                                            # delmas_dir_gain,
                                            # delmas_dir_li,
                                            cedara_dir_mrnn,
                                            # cedara_dir_gain,
                                            # cedara_dir_li,
                                        ]:
                                            farm_id = "delmas"
                                            if "cedara" in str(dataset).lower():
                                                farm_id = "cedara"
                                            main_experiment.main(
                                                output_dir=output_dir
                                                / "main_experiment"
                                                / clf
                                                / dataset.stem
                                                / f"{dataset.stem}_{farm_id}_{cv}_{i_day}_{a_day}_{w_day}_{slug}_season_{add_seasons_to_features}"
                                                / class_unhealthy_label,
                                                dataset_folder=dataset,
                                                preprocessing_steps=steps,
                                                n_imputed_days=i_day,
                                                n_activity_days=a_day,
                                                n_weather_days=w_day,
                                                cv=cv,
                                                classifiers=[clf],
                                                class_unhealthy_label=[
                                                    class_unhealthy_label
                                                ],
                                                study_id=farm_id,
                                                add_seasons_to_features=add_seasons_to_features,
                                                export_fig_as_pdf=False,
                                                plot_2d_space=False,
                                                pre_visu=False,
                                                export_hpc_string = export_hpc_string,
                                                skip=False,
                                                weather_file=Path(
                                                    "C:/Users/fo18103/PycharmProjects/PredictionOfHelminthsInfection/weather_data/delmas_south_africa_2011-01-01_to_2015-12-31.csv"
                                                ),
                                                n_job=n_job,
                                            )

                                # for dataset in [cedara_dir_gain, cedara_dir_li, cedara_dir_mrnn]:
                                #     main_experiment.main(
                                #         output_dir=output_dir
                                #         / "main_experiment"
                                #         / f"cedara_{dataset.stem}_{cv}_{i_day}_{a_day}_{w_day}_{slug}_season_{add_seasons_to_features}"
                                #         / "2To2",
                                #         dataset_folder=dataset,
                                #         preprocessing_steps=steps,
                                #         n_imputed_days=i_day,
                                #         n_activity_days=a_day,
                                #         n_weather_days=w_day,
                                #         cv=cv,
                                #         classifiers=["rbf"],
                                #         class_unhealthy_label=["2To2"],
                                #         study_id="cedara",
                                #         add_seasons_to_features=add_seasons_to_features,
                                #         export_fig_as_pdf=False,
                                #         plot_2d_space=True,
                                #         weather_file=Path(
                                #             "C:/Users/fo18103/PycharmProjects/PredictionOfHelminthsInfection/weather_data/cedara_south_africa_2011-01-01_to_2015-12-31.csv"),
                                #     )
                                # main_experiment.main(
                                #     output_dir=output_dir
                                #     / "main_experiment"
                                #     / f"cedara_{cv}_{i_day}_{a_day}_{w_day}_{slug}_season_{add_seasons_to_features}"
                                #     / "2To2",
                                #     dataset_folder=cedara_dir,
                                #     preprocessing_steps=steps,
                                #     n_imputed_days=i_day,
                                #     n_activity_days=a_day,
                                #     n_weather_days=w_day,
                                #     class_unhealthy_label=["2To2"],
                                #     cv=cv,
                                #     classifiers=["linear", "rbf"],
                                #     study_id="cedara",
                                #     add_seasons_to_features=add_seasons_to_features,
                                #     plot_2d_space=False,
                                #     pre_visu=False,
                                #     export_fig_as_pdf=True,
                                #     weather_file=Path(
                                #         "C:/Users/fo18103/PycharmProjects/PredictionOfHelminthsInfection/weather_data/cedara_south_africa_2011-01-01_to_2015-12-31.csv"),
                                # )
                                # continue

                                # # main_experiment.main(
                                # #     output_dir=output_dir
                                # #     / "main_experiment"
                                # #     / f"delmas_{cv}_{i_day}_{a_day}_{slug}_season_{add_seasons_to_features}"
                                # #     / "1To1_2To1__2To2",
                                # #     dataset_folder=delmas_dir,
                                # #     preprocessing_steps=steps,
                                # #     n_imputed_days=i_day,
                                # #     n_activity_days=a_day,
                                # #     cv=cv,
                                # #     classifiers=["linear", "rbf"],
                                # #     class_healthy_label=["1To1", "2To1"],
                                # #     class_unhealthy_label=["2To2"],
                                # #     study_id="delmas",
                                # #     add_seasons_to_features=add_seasons_to_features,
                                # # )
                                # #
                                # # main_experiment.main(
                                # #     output_dir=output_dir
                                # #     / "main_experiment"
                                # #     / f"delmas_{cv}_{i_day}_{a_day}_{slug}_season_{add_seasons_to_features}"
                                # #     / "1To1_2To1__2To2_1To2",
                                # #     dataset_folder=delmas_dir,
                                # #     preprocessing_steps=steps,
                                # #     n_imputed_days=i_day,
                                # #     n_activity_days=a_day,
                                # #     cv=cv,
                                # #     classifiers=["linear", "rbf"],
                                # #     class_healthy_label=["1To1", "2To1"],
                                # #     class_unhealthy_label=["2To2", "1To2"],
                                # #     study_id="delmas",
                                # #     add_seasons_to_features=add_seasons_to_features,
                                # # )
                                # #
                                # # main_experiment.main(
                                # #     output_dir=output_dir
                                # #     / "main_experiment"
                                # #     / f"delmas_{cv}_{i_day}_{a_day}_{slug}_season_{add_seasons_to_features}"
                                # #     / "1To2",
                                # #     dataset_folder=delmas_dir,
                                # #     preprocessing_steps=steps,
                                # #     n_imputed_days=i_day,
                                # #     n_activity_days=a_day,
                                # #     class_unhealthy_label=["1To2"],
                                # #     cv=cv,
                                # #     classifiers=["linear", "rbf"],
                                # #     study_id="delmas",
                                # #     add_seasons_to_features=add_seasons_to_features
                                # # )
                                #
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
                                #     add_seasons_to_features=add_seasons_to_features,
                                #     plot_2d_space=True,
                                #     export_fig_as_pdf=True
                                # )
                                #
                                # main_experiment.main(
                                #     output_dir=output_dir
                                #     / "main_experiment"
                                #     / f"cedara_{cv}_{i_day}_{a_day}_{w_day}_{slug}_season_{add_seasons_to_features}"
                                #     / "2To1",
                                #     dataset_folder=cedara_dir_mrnn,
                                #     preprocessing_steps=steps,
                                #     n_imputed_days=i_day,
                                #     n_activity_days=a_day,
                                #     class_unhealthy_label=["2To1"],
                                #     cv=cv,
                                #     classifiers=["linear", "rbf"],
                                #     study_id="cedara",
                                #     add_seasons_to_features=add_seasons_to_features,
                                #     plot_2d_space=False,
                                #     export_fig_as_pdf=True,
                                #     pre_visu=False
                                # )
                                #
                                #
                                # main_experiment.main(
                                #     output_dir=output_dir
                                #     / "main_experiment"
                                #     / f"cedara_{cv}_{i_day}_{a_day}_{slug}_season_{add_seasons_to_features}"
                                #     / "2To2_2To4_3To4_1To4_1To3_4To5_2To3",
                                #     dataset_folder=cedara_dir,
                                #     preprocessing_steps=steps,
                                #     n_imputed_days=i_day,
                                #     n_activity_days=a_day,
                                #     class_unhealthy_label=["2To2", "2To4", "3To4", "1To4", "1To3", "4To5", "2To3"],
                                #     cv=cv,
                                #     classifiers=["linear", "rbf"],
                                #     study_id="cedara",
                                #     add_seasons_to_features=add_seasons_to_features,
                                #     plot_2d_space=True,
                                #     export_fig_as_pdf=True
                                # )

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
                                #     n_weather_days=w_day,
                                #     cv=cv,
                                #     classifiers=["linear", "rbf"],
                                #     study_id="cedara",
                                #     add_seasons_to_features=add_seasons_to_features,
                                #     plot_2d_space=True,
                                #     export_fig_as_pdf=True,
                                #     weather_file=Path(
                                #         "C:/Users/fo18103/PycharmProjects/PredictionOfHelminthsInfection/weather_data/src/cedara_weather_raw.json"),
                                # )
                                # continue

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

        i = 7
        n_a = 6
        temporal_validation.main(
            output_dir=output_dir
            / "temporal_validation"
            / f"delmas_{i}_{n_a}"
            / "2To2",
            dataset_folder=delmas_dir_mrnn,
            n_imputed_days=i,
            n_activity_days=n_a,
            export_fig_as_pdf=True,
        )

        # i = 7
        # n_a = 6
        # temporal_validation.main(
        #     output_dir=output_dir / "temporal_validation" / f"cedara_{i}_{n_a}" / "2To2",
        #     dataset_folder=cedara_dir,
        #     n_imputed_days=i,
        #     n_activity_days=n_a,
        #     sample_date_filter="2013-02-14",
        #     class_healthy_label=["1To1"],
        #     class_unhealthy_label=["2To2", "2To4", "3To4", "1To4", "1To3", "4To5", "2To3"],
        #     export_fig_as_pdf=True)

    if exp_cross_farm:
        print("experiment 3: cross farm validation")
        for imp_d in [7]:
            for a_act_day in [1, 4, 7]:
                cross_farm_validation.main(
                    farm1_path=delmas_dir_mrnn,
                    farm2_path=cedara_dir_mrnn,
                    output_dir=output_dir
                    / "cross_farm"
                    / f"{imp_d}_{a_act_day}"
                    / "2To2",
                    n_imputed_days=imp_d,
                    n_activity_days=a_act_day,
                    class_unhealthy_f2=[
                        "2To2"
                    ]
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


def purge_hpc_file(filename):
    if os.path.exists(filename):
        os.remove(filename)


if __name__ == "__main__":
    #single_run(dataset=Path("E:/thesis/datasets/cedara/cedara_datasetmrnn7_23"), farm_id="cedara", export_hpc_string=False)
    #single_run(dataset=Path("E:/thesis/datasets/delmas/delmas_dataset4_mrnn_7day"), farm_id="delmas", export_hpc_string=False)
    # local_run()
    # single_run(dataset=Path("E:/thesis/datasets/delmas/delmas_dataset_mrnn_30days"), farm_id="delmas")
    #single_run(dataset=Path("E:/thesis/datasets/cedara/cedara_dataset_mrnn_30days"), farm_id="cedara")

    #single_run(dataset=Path("E:/thesis/datasets/delmas/delmas_dataset4_mrnn_7day"), farm_id="delmas")
    # single_run(dataset=Path("E:/thesis/datasets/delmas/delmas_dataset_1_gain_66_no_filter_fixed"), farm_id="delmas")
    # single_run(dataset=Path("E:/thesis/datasets/delmas/delmas_dataset_1_li_66_no_filter_fixed"), farm_id="delmas")
    # #
    # # single_run(dataset=Path("E:/thesis/datasets/cedara/cedara_datasetmrnn7_23"), farm_id="cedara")
    # single_run(dataset=Path("E:/thesis/datasets/cedara/cedara_dataset_1_gain_172_no_filter_fixed"), farm_id="cedara")
    # single_run(dataset=Path("E:/thesis/datasets/cedara/cedara_dataset_1_li_172_no_filter_fixed"), farm_id="cedara")

    #single_run(dataset=Path("E:/thesis/datasets/delmas/delmas_dataset4_mrnn_7day"), farm_id="delmas")
    #single_run(dataset=Path("E:/thesis/datasets/cedara/cedara_datasetmrnn7_23"), farm_id="cedara")
    #single_run(dataset=Path("E:/thesis/datasets/cedara/cedara_dataset_li_7_23"), farm_id="cedara")
    #single_run(dataset=Path("E:/thesis/datasets/cedara/cedara_dataset_1_gain_172"), farm_id="cedara")

    biospi_run()
    # typer.run(main)
