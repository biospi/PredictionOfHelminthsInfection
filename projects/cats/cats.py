import typer
import ml as main_pipeline
from pathlib import Path


def main():
    """Thesis script runs the cats study
    Args:\n
        output_dir: Output directory
    """
    for clf in ["rbf", "cnn"]:
        for steps in [
            ["LINEAR", "QN", "CWT(MEXH)", "STD"],
            ["LINEAR", "QN", "CWT(MORL)", "STD"],
            ["LINEAR", "QN", "CWT(MEXH)"],
            ["LINEAR", "QN", "CWT(MORL)"],
            ["LINEAR", "QN", "CENTER", "CWT(MEXH)", "STD"],
            ["LINEAR", "QN", "CENTER", "CWT(MORL)", "STD"],
            ["LINEAR", "QN", "CENTER", "CWT(MEXH)"],
            ["LINEAR", "QN", "CENTER", "CWT(MORL)"],
            ["LINEAR", "QN", "STD"],
            ["LINEAR", "QN", "STD", "CWT(MORL)"],
            ["LINEAR", "QN", "STD", "CENTER", "CWT(MORL)"],
            ["LINEAR", "QN", "ANSCOMBE", "LOG"],
            ["LINEAR", "QN", "ANSCOMBE", "LOG", "CENTER", "CWT(MORL)"],
            ["LINEAR", "QN", "ANSCOMBE", "LOG", "CWT(MORL)"],
            ["LINEAR", "QN", "ANSCOMBE", "LOG", "STD"],
            ["LINEAR", "QN", "ANSCOMBE", "LOG", "CENTER", "CWT", "STD"],
            ["LINEAR", "QN", "ANSCOMBE", "CENTER", "CWT(MORL)"],
            ["LINEAR", "QN", "ANSCOMBE", "STD"],
            ["LINEAR", "QN", "ANSCOMBE", "CENTER", "CWT(MORL)", "STD"],
            ["LINEAR", "QN", "LOG", "CENTER", "CWT(MORL)"],
            ["LINEAR", "QN", "LOG", "STD"],
            ["LINEAR", "QN", "LOG", "CENTER", "CWT(MORL)", "STD"],
        ]:
            slug = "_".join(steps)

            for cv in ["StratifiedLeaveTwoOut", "RepeatedKFold"]:
                main_pipeline.main(
                    output_dir=Path(f"E:/Cats/ml_peak/build_sec_10_rois_2/rois/{clf}/{slug}_{cv}"),
                    dataset_folder=Path(
                        f"E:/Cats/build_sec_10_rois_2/dataset/training_sets/day_w"
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
                    meta_col_str=["name", "age", "mobility_score"],
                    classifiers=[clf],
                    n_imputed_days=-1,
                    n_activity_days=-1,
                    class_healthy_label=["0.0"],
                    class_unhealthy_label=["1.0"],
                    n_scales=6,
                    n_splits=3,
                    n_repeats=4,
                    n_job=7,
                    study_id="cat",
                    cv=cv,
                    output_qn_graph=True,
                    epoch=500,
                )

                main_pipeline.main(
                    output_dir=Path(f"E:/Cats/ml_peak/build_sec_10_rois_3/rois/{clf}/{slug}_{cv}"),
                    dataset_folder=Path(
                        f"E:/Cats/build_sec_10_rois_3/dataset/training_sets/day_w"
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
                    meta_col_str=["name", "age", "mobility_score"],
                    classifiers=[clf],
                    n_imputed_days=-1,
                    n_activity_days=-1,
                    class_healthy_label=["0.0"],
                    class_unhealthy_label=["1.0"],
                    n_scales=6,
                    n_splits=3,
                    n_repeats=4,
                    n_job=7,
                    study_id="cat",
                    cv=cv,
                    output_qn_graph=True,
                    epoch=500,
                )


if __name__ == "__main__":
    typer.run(main)
