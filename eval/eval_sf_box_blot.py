from utils.visualisation import plot_ml_report_final
from pathlib import Path


def main(path):
    plot_ml_report_final(Path(path))


if __name__ == "__main__":
    main('H:/thesis_final_feb16/thesis_final_feb16/main_experiment/rbf/delmas_dataset4_mrnn_7day')