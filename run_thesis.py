import typer
import ml as main_experiment
import ml_cross_farm_validation as cross_farm_validation
import ml_temporal_validation as temporal_validation
from pathlib import Path


def main(
    exp_main: bool = False,
    exp_temporal: bool = True,
    exp_cross_farm: bool = True,
    output_dir: Path = Path("E:/thesis2"),
):
    """Thesis script run all key experiments for data exploration chapter
    Args:\n
        output_dir: Output directory
    """

    if exp_main:
        print("experiment 1: main pipeline")

        steps_list = [["QN", "ANSCOMBE", "LOG"]]
        for steps in steps_list:
            slug = "_".join(steps)

            for i_day in [7, 6]:
                for a_day in [7, 1]:
                    for cv in ['RepeatedKFold']:
                        for add_seasons_to_features in [False, True]:

                            main_experiment.main(
                                output_dir=output_dir
                                / "main_experiment"
                                / f"delmas_{cv}_{i_day}_{a_day}_{slug}_season_{add_seasons_to_features}"
                                / "2To2",
                                dataset_folder=Path("E:/Data2/debug3/delmas/dataset4_mrnn_7day"),
                                preprocessing_steps=steps,
                                n_imputed_days=i_day,
                                n_activity_days=a_day,
                                cv=cv,
                                study_id="delmas",
                                add_seasons_to_features=add_seasons_to_features
                            )

                            main_experiment.main(
                                output_dir=output_dir
                                / "main_experiment"
                                / f"cedara_{cv}_{i_day}_{a_day}_{slug}_season_{add_seasons_to_features}"
                                / "2To2",
                                dataset_folder=Path("E:/Data2/debug3/cedara/dataset6_mrnn_7day"),
                                preprocessing_steps=steps,
                                n_imputed_days=i_day,
                                n_activity_days=a_day,
                                class_unhealthy_label=["2To2"],
                                cv=cv,
                                study_id="cedara",
                                add_seasons_to_features=add_seasons_to_features
                            )
                            # main_experiment.main(
                            #     output_dir=output_dir
                            #     / "main_experiment"
                            #     / f"cedara_{cv}_{i_day}_{a_day}_{slug}"
                            #     / "4To4_3To5_4To3_5To3_2To5_2To2",
                            #     dataset_folder=Path("E:/Data2/debug3/cedara/dataset6_mrnn_7day"),
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
                            #     farm_id="cedara"
                            # )

                            main_experiment.main(
                                output_dir=output_dir
                                / "main_experiment"
                                / f"delmas_{cv}_{i_day}_{a_day}_{slug}_season_{add_seasons_to_features}"
                                / "1To2",
                                dataset_folder=Path("E:/Data2/debug3/delmas/dataset4_mrnn_7day"),
                                preprocessing_steps=steps,
                                n_imputed_days=i_day,
                                n_activity_days=a_day,
                                class_unhealthy_label=["1To2"],
                                cv=cv,
                                study_id="delmas",
                                add_seasons_to_features=add_seasons_to_features
                            )

                            main_experiment.main(
                                output_dir=output_dir
                                / "main_experiment"
                                / f"cedara_{cv}_{i_day}_{a_day}_{slug}_season_{add_seasons_to_features}"
                                / "1To2",
                                dataset_folder=Path("E:/Data2/debug3/cedara/dataset6_mrnn_7day"),
                                preprocessing_steps=steps,
                                n_imputed_days=i_day,
                                n_activity_days=a_day,
                                class_unhealthy_label=["1To2"],
                                cv=cv,
                                study_id="cedara",
                                add_seasons_to_features=add_seasons_to_features
                            )

                            main_experiment.main(
                                output_dir=output_dir
                                / "main_experiment"
                                / f"delmas_{cv}_{i_day}_{a_day}_{slug}_season_{add_seasons_to_features}"
                                / "2To1",
                                dataset_folder=Path("E:/Data2/debug3/delmas/dataset4_mrnn_7day"),
                                preprocessing_steps=steps,
                                n_imputed_days=i_day,
                                n_activity_days=a_day,
                                class_unhealthy_label=["2To1"],
                                cv=cv,
                                study_id="delmas",
                                add_seasons_to_features=add_seasons_to_features
                            )

                            main_experiment.main(
                                output_dir=output_dir
                                / "main_experiment"
                                / f"cedara_{cv}_{i_day}_{a_day}_{slug}_season_{add_seasons_to_features}"
                                / "2To1_3To1",
                                dataset_folder=Path("E:/Data2/debug3/cedara/dataset6_mrnn_7day"),
                                preprocessing_steps=steps,
                                n_imputed_days=i_day,
                                n_activity_days=a_day,
                                class_unhealthy_label=["2To1", "3To1"],
                                cv=cv,
                                study_id="cedara",
                                add_seasons_to_features=add_seasons_to_features
                            )

    if exp_temporal:
        print("experiment 2: temporal validation")
        for i in [7, 6]:
            temporal_validation.main(
                output_dir=output_dir / "temporal_validation" / f"delmas_{i}" / "2To2",
                dataset_folder=Path("E:/Data2/debug/delmas/dataset_mrnn_7day"),
                n_imputed_days=i,
            )

            # temporal_validation.main(
            #     output_dir=output_dir
            #     / "temporal_validation"
            #     / f"cedara_{i}"
            #     / "4To4_3To5_4To3_5To3_2To5_2To2",
            #     dataset_folder=Path("E:/Data2/debug3/cedara/dataset6_mrnn_7day"),
            #     n_imputed_days=i,
            #     class_unhealthy_label=[
            #         "4To4",
            #         "3To5",
            #         "4To3",
            #         "5To3",
            #         "2To5",
            #         "2To2",
            #     ],
            # )

            temporal_validation.main(
                output_dir=output_dir / "temporal_validation" / f"cedara_{i}" / "2To2",
                dataset_folder=Path("E:/Data2/debug3/cedara/dataset6_mrnn_7day"),
                n_imputed_days=i,
            )

    if exp_cross_farm:
        print("experiment 3: cross farm validation")
        for imp_d in [7, 6]:
            for a_act_day in [7]:
                cross_farm_validation.main(
                    farm1_path=Path("E:\Data2\debug3\delmas\dataset4_mrnn_7day"),
                    farm2_path=Path("E:\Data2\debug3\cedara\dataset6_mrnn_7day"),
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
    typer.run(main)
