from utils.visualisation import plot_ml_report_final
from pathlib import Path


def main(path):
    plot_ml_report_final(Path(path), filter_for_norm=False)
    #plot_ml_report_final(Path(path), filter_per_clf=False)


if __name__ == "__main__":
    main('H:/thesis_final_feb16/thesis_final_feb16/main_experiment')