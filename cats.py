import typer
import ml as main_pipeline
from pathlib import Path

from utils.Utils import purge_hpc_file


# nohup python3 cats.py --dataset-parent /mnt/storage/scratch/axel/cats/peak --out-parent /mnt/storage/scratch/axel/cats/ml_peak > peak.log &


def main(
    out_parent: str = "/user/work/fo18103/Cats/ml_build_permutations_thesis_rev",
    dataset_parent: str = "/user/work/fo18103/Cats/build_permutations_final",
    # out_parent: str = "E:/Cats/ml_build_permutations_thesis_rev",
    # dataset_parent: str = "E:/Cats/build_permutations_final",
    export_hpc_string: bool = True,
    biospi_run: bool = False,
    n_job: int = 28,
):
    """Thesis script runs the cats study
    Args:\n
        out_parent: Output directory
        dataset_parent: Dataset directory
    """
    if biospi_run:
        dataset_parent = "/mnt/storage/scratch/axel/cats/build_permutations_final"
        out_parent = "/mnt/storage/scratch/axel/cats/ml"

    print(out_parent)
    print(dataset_parent)
    print(export_hpc_string)

    if export_hpc_string:
        purge_hpc_file('thesis_hpc_ln.txt')
        purge_hpc_file('thesis_hpc.txt')

    for clf in ["rbf"]:
        for steps in [
            ["QN"],
            ["QN", "STD"]
        ]:
            slug = "_".join(steps)
            print(slug)
            folders = sorted([x.stem for x in Path(dataset_parent).glob("*")])
            folders = [x for x in folders if "visu" not in str(x)]
            # folders = ["800__001__0_00100__120" "1000__002__0_00100__120", "1000__003__0_00100__120", "5000__004__0_00100__120"]
            folders = ["1000__006__0_00100__030", "1000__006__0_00100__060", "1000__005__0_00100__120"]
            #folders = ["800__001__0_00100__120"]
            # folders = [
            #     "800__001__0_00100__120",
            #     "800__002__0_00100__120",
            #     "800__003__0_00100__120",
            #     "800__004__0_00100__120",
            #     "800__006__0_00100__120",
            #     "1000__001__0_00100__120",
            #     "1000__002__0_00100__120",
            #     "1000__003__0_00100__120",
            #     "1000__004__0_00100__120",
            #     "1000__006__0_00100__120",
            #     "2000__001__0_00100__120",
            #     "2000__002__0_00100__120",
            #     "2000__003__0_00100__120",
            #     "2000__004__0_00100__120",
            #     "2000__006__0_00100__120",
            #     "3000__004__0_00100__120",
            #     "4000__004__0_00100__120",
            #     "5000__001__0_00100__120",
            #     "5000__002__0_00100__120",
            #     "5000__003__0_00100__120",
            #     "5000__004__0_00100__120",
            #     "5000__006__0_00100__120",
            #     "6000__004__0_00100__120",
            # ]

            #folders = ["1000__002__0_00100__120"]
            print(folders)
            for thresh in folders:
                print(f"threshold={thresh}")
                for cv in ["LeaveOneOut"]:
                    main_pipeline.main(
                        output_dir=Path(f"{out_parent}/{thresh}/{clf}/{slug}_{cv}"),
                        dataset_folder=Path(
                            f"{dataset_parent}/{thresh}/dataset/training_sets/samples"
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
                        individual_to_keep=[],
                        # individual_to_test=[
                        #     603,
                        #     627,
                        #     77,
                        #     651,
                        #     607,
                        #     661,
                        #     621,
                        #     609,
                        #     632,
                        #     658,
                        # ],
                        classifiers=[clf],
                        n_imputed_days=-1,
                        n_activity_days=-1,
                        class_healthy_label=["0.0"],
                        class_unhealthy_label=["1.0"],
                        n_scales=8,
                        n_splits=5,
                        n_repeats=10,
                        n_job=n_job,
                        study_id="cat",
                        cv=cv,
                        output_qn_graph=False,
                        enable_qn_peak_filter=False,
                        pre_visu=False,
                        epoch=100,
                        batch_size=100,
                        plot_2d_space=False,
                        export_hpc_string=export_hpc_string
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
    main()
    # typer.run(main)
