import typer
import ml as main_pipeline
from pathlib import Path


def main():
    """Thesis script runs the cats study
    Args:\n
        output_dir: Output directory
    """
    for steps in [["LINEAR", "QN", "ANSCOMBE", "LOG"]]:
        slug = "_".join(steps)

        for f in [[]]:
            main_pipeline.main(
                output_dir=Path(f"E:/Cats/ml/build_min_1440_1440/day_w/{f}/{slug}"),
                dataset_folder=Path(f"E:/Cats/build_min_1440_1440/dataset/training_sets/day_w"),
                preprocessing_steps=steps,
                meta_columns=["label", "id", "imputed_days", "date", "health", "target", "age", "name", "mobility_score"],
                meta_col_str=["name", "age", "mobility_score"],
                classifiers=["cnn"],
                add_feature=f,
                n_imputed_days=-1,
                n_activity_days=-1,
                class_healthy_label=["0.0"],
                class_unhealthy_label=["1.0"],
                n_splits=2,
                n_repeats=2,
                n_job=1,
                study_id="cat",
                cv="RepeatedKFold",
                epoch=3
            )

    # for f in [[], ["age"], ["mobility_score"]]:
    #     main(
    #         output_dir=Path(f"E:/Cats/ml/build_min_1440_720/day_w/{f}/{slug}"),
    #         dataset_folder=Path(f"E:/Cats/build_min_1440_720/dataset/training_sets/day_w"),
    #         preprocessing_steps=steps,
    #         meta_columns=["label", "id", "imputed_days", "date", "health", "target", "age", "name", "mobility_score"],
    #         meta_col_str=["name", "age", "mobility_score"],
    #         svc_kernel=["rbf"],
    #         add_feature=f,
    #         n_imputed_days=-1,
    #         n_activity_days=-1,
    #         class_healthy_label=["0.0"],
    #         class_unhealthy_label=["1.0"],
    #         n_splits=5,
    #         n_repeats=10,
    #         n_job=5,
    #         study_id="cat",
    #         cv="RepeatedStratifiedKFold"
    #     )


if __name__ == "__main__":
    typer.run(main)