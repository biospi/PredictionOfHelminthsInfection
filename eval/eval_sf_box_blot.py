from utils.visualisation import plot_ml_report_final
from pathlib import Path


def main(path):
    plot_ml_report_final(Path(path), filter_per_clf=False)


if __name__ == "__main__":
    main('E:/thesis_debug_weather7/main_experiment')