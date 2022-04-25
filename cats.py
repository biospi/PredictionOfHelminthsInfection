import typer
import ml as main_pipeline
from pathlib import Path

#nohup python3 cats.py --dataset-parent /mnt/storage/scratch/axel/cats/peak --out-parent /mnt/storage/scratch/axel/cats/ml_peak > peak.log &


def main(
    out_parent: str = "E:/Cats/ml_peak_build_sec_w4min2_std",
    dataset_parent: str = "E:/Cats/build_sec2/peak",
):
    """Thesis script runs the cats study
    Args:\n
        out_parent: Output directory
        dataset_parent: Dataset directory
    """
    print(out_parent)
    print(dataset_parent)

    for clf in ["rbf"]:
        for steps in [
            # ["LINEAR", "QN", "STD"],
            # ["LINEAR", "QN", "ANSCOMBE", "LOG", "STD"]
            # ["LINEAR", "QN", "LOG", "STD"],
            # ["LINEAR", "QN", "ANSCOMBE", "LOG"],
            # ["LINEAR", "QN", "ANSCOMBE", "LOG", "STD"],
            # ["LINEAR", "QN", "CWT(MEXH)", "STD"],
            # ["LINEAR", "QN", "CWT(MORL)", "STD"],
            # ["LINEAR", "QN", "CWT(MEXH)"],
            # ["LINEAR", "QN", "CWT(MORL)"],
            # ["LINEAR", "QN", "CENTER", "CWT(MEXH)", "STD"],
            # ["LINEAR", "QN", "CENTER", "CWT(MORL)", "STD"],
            # ["LINEAR", "QN", "CENTER", "CWT(MEXH)"],
            # ["LINEAR", "QN", "CENTER", "CWT(MORL)"],
            # ["LINEAR", "QN", "STD", "CWT(MORL)"],
            # ["LINEAR", "QN", "STD", "CENTER", "CWT(MORL)"],
            ["LINEAR", "QN", "ANSCOMBE", "LOG", "CENTER", "CWT(MORL)", "STDSCALE"],
            # ["LINEAR", "QN", "ANSCOMBE", "LOG", "CWT(MORL)"],
            # ["LINEAR", "QN", "ANSCOMBE", "LOG", "CENTER", "CWT", "STD"],
            # ["LINEAR", "QN", "ANSCOMBE", "CENTER", "CWT(MORL)"],
            # ["LINEAR", "QN", "ANSCOMBE", "CENTER", "CWT(MORL)", "STD"],
            # ["LINEAR", "QN", "LOG", "CENTER", "CWT(MORL)"],
            # ["LINEAR", "QN", "LOG", "CENTER", "CWT(MORL)", "STD"],
        ]:
            slug = "_".join(steps)
            print(slug)
            folders = [x.stem for x in Path(dataset_parent).glob("*")]
            print(folders)
            for thresh in folders:
                print(f"threshold={thresh}")
                for cv in ["RepeatedKFold", "StratifiedLeaveTwoOut"]:
                    main_pipeline.main(
                        output_dir=Path(f"{out_parent}/{thresh}/{clf}/{slug}_{cv}"),
                        dataset_folder=Path(
                            f"{dataset_parent}/{thresh}/dataset/training_sets/day_w"
                        ),
                        preprocessing_steps=steps,
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
                        meta_col_str=[],
                        individual_to_ignore=["MrDudley", "Oliver_F", "Lucy"],
                        classifiers=[clf],
                        n_imputed_days=-1,
                        n_activity_days=-1,
                        class_healthy_label=["0.0"],
                        class_unhealthy_label=["1.0"],
                        n_scales=8,
                        n_splits=5,
                        n_repeats=10,
                        n_job=7,
                        study_id="cat",
                        cv=cv,
                        output_qn_graph=True,
                        epoch=500,
                    )

                # main_pipeline.main(
                #     output_dir=Path(
                #         f"{parent}/build_sec_10_rois_001/rois/{clf}/{slug}_{cv}"
                #     ),
                #     dataset_folder=Path(
                #         f"E:/Cats/build_sec_10_rois_001/dataset/training_sets/day_w"
                #     ),
                #     preprocessing_steps=steps,
                #     meta_columns=[
                #         "label",
                #         "id",
                #         "imputed_days",
                #         "date",
                #         "health",
                #         "target",
                #         "age",
                #         "name",
                #         "mobility_score",
                #     ],
                #     meta_col_str=["name", "age", "mobility_score"],
                #     individual_to_ignore=["MrDudley", "Oliver_F", "Lucy"],
                #     classifiers=[clf],
                #     n_imputed_days=-1,
                #     n_activity_days=-1,
                #     class_healthy_label=["0.0"],
                #     class_unhealthy_label=["1.0"],
                #     n_scales=6,
                #     n_splits=5,
                #     n_repeats=10,
                #     n_job=7,
                #     study_id="cat",
                #     cv=cv,
                #     output_qn_graph=True,
                #     epoch=500,
                # )

                # main_pipeline.main(
                #     output_dir=Path(
                #         f"{parent}/build_sec_10_rois_25/rois/{clf}/{slug}_{cv}"
                #     ),
                #     dataset_folder=Path(
                #         f"E:/Cats/build_sec_10_rois_25/dataset/training_sets/day_w"
                #     ),
                #     preprocessing_steps=steps,
                #     meta_columns=[
                #         "label",
                #         "id",
                #         "imputed_days",
                #         "date",
                #         "health",
                #         "target",
                #         "age",
                #         "name",
                #         "mobility_score",
                #     ],
                #     meta_col_str=["name", "age", "mobility_score"],
                #     individual_to_ignore=["MrDudley", "Oliver_F", "Lucy"],
                #     classifiers=[clf],
                #     n_imputed_days=-1,
                #     n_activity_days=-1,
                #     class_healthy_label=["0.0"],
                #     class_unhealthy_label=["1.0"],
                #     n_scales=7,
                #     n_splits=5,
                #     n_repeats=10,
                #     n_job=7,
                #     study_id="cat",
                #     cv=cv,
                #     output_qn_graph=True,
                #     epoch=500,
                # )

                # main_pipeline.main(
                #     output_dir=Path(
                #         f"{parent}/build_sec_10_rois_0001/rois/{clf}/{slug}_{cv}"
                #     ),
                #     dataset_folder=Path(
                #         f"E:/Cats/build_sec_10_rois_0001/dataset/training_sets/day_w"
                #     ),
                #     preprocessing_steps=steps,
                #     meta_columns=[
                #         "label",
                #         "id",
                #         "imputed_days",
                #         "date",
                #         "health",
                #         "target",
                #         "age",
                #         "name",
                #         "mobility_score",
                #     ],
                #     meta_col_str=["name", "age", "mobility_score"],
                #     individual_to_ignore=["MrDudley", "Oliver_F", "Lucy"],
                #     classifiers=[clf],
                #     n_imputed_days=-1,
                #     n_activity_days=-1,
                #     class_healthy_label=["0.0"],
                #     class_unhealthy_label=["1.0"],
                #     n_scales=7,
                #     n_splits=5,
                #     n_repeats=10,
                #     n_job=7,
                #     study_id="cat",
                #     cv=cv,
                #     output_qn_graph=True,
                #     epoch=500,
                # )

                # main_pipeline.main(
                #     output_dir=Path(
                #         f"{parent}/build_sec_10_rois_00001/rois/{clf}/{slug}_{cv}"
                #     ),
                #     dataset_folder=Path(
                #         f"E:/Cats/build_sec_10_rois_00001/dataset/training_sets/day_w"
                #     ),
                #     preprocessing_steps=steps,
                #     meta_columns=[
                #         "label",
                #         "id",
                #         "imputed_days",
                #         "date",
                #         "health",
                #         "target",
                #         "age",
                #         "name",
                #         "mobility_score",
                #     ],
                #     meta_col_str=["name", "age", "mobility_score"],
                #     individual_to_ignore=["MrDudley", "Oliver_F", "Lucy"],
                #     classifiers=[clf],
                #     n_imputed_days=-1,
                #     n_activity_days=-1,
                #     class_healthy_label=["0.0"],
                #     class_unhealthy_label=["1.0"],
                #     n_scales=6,
                #     n_splits=5,
                #     n_repeats=10,
                #     n_job=7,
                #     study_id="cat",
                #     cv=cv,
                #     output_qn_graph=True,
                #     epoch=500,
                # )

                # main_pipeline.main(
                #     output_dir=Path(f"{parent}/build_sec_10_rois_25/rois/{clf}/{slug}_{cv}"),
                #     dataset_folder=Path(
                #         f"E:/Cats/build_sec_10_rois_25/dataset/training_sets/day_w"
                #     ),
                #     preprocessing_steps=steps,
                #     meta_columns=[
                #         "label",
                #         "id",
                #         "imputed_days",
                #         "date",
                #         "health",
                #         "target",
                #         "age",
                #         "name",
                #         "mobility_score",
                #     ],
                #     meta_col_str=["name", "age", "mobility_score"],
                #     individual_to_ignore=["MrDudley", "Oliver_F", "Lucy"],
                #     classifiers=[clf],
                #     n_imputed_days=-1,
                #     n_activity_days=-1,
                #     class_healthy_label=["0.0"],
                #     class_unhealthy_label=["1.0"],
                #     n_scales=6,
                #     n_splits=3,
                #     n_repeats=4,
                #     n_job=7,
                #     study_id="cat",
                #     cv=cv,
                #     output_qn_graph=True,
                #     epoch=500,
                # )

                # main_pipeline.main(
                #     output_dir=Path(f"E:/Cats/ml_peak/build_sec_10_rois_3/rois/{clf}/{slug}_{cv}"),
                #     dataset_folder=Path(
                #         f"E:/Cats/build_sec_10_rois_3/dataset/training_sets/day_w"
                #     ),
                #     preprocessing_steps=steps,
                #     meta_columns=[
                #         "label",
                #         "id",
                #         "imputed_days",
                #         "date",
                #         "health",
                #         "target",
                #         "age",
                #         "name",
                #         "mobility_score",
                #     ],
                #     meta_col_str=["name", "age", "mobility_score"],
                #     individual_to_ignore=["MrDudley", "Oliver_F", "Lucy"],
                #     classifiers=[clf],
                #     n_imputed_days=-1,
                #     n_activity_days=-1,
                #     class_healthy_label=["0.0"],
                #     class_unhealthy_label=["1.0"],
                #     n_scales=6,
                #     n_splits=3,
                #     n_repeats=4,
                #     n_job=7,
                #     study_id="cat",
                #     cv=cv,
                #     output_qn_graph=True,
                #     epoch=500,
                # )


if __name__ == "__main__":
    typer.run(main)
