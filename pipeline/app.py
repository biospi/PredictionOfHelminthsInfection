import subprocess

if __name__ == "__main__":
    print("start...")
    threshi = 15
    threshz = 720
    ndays = 2
    print("***********************************************************************************************************")
    subprocess.call(
        "python interpolation_zero2nan_thresh.py F:/Data2/thresholded F:/Data/backfill_1min/delmas_70101200027 %i %i 0 6" % (
            threshi, threshz), shell=True)
    print("***********************************************************************************************************")
    subprocess.call(
        "python gen_training_sets.py F:/Data2/gen_dataset_debug F:/Data2/thresholded/delmas_70101200027_interpol_%d_zeros_%d_imputation_0 F:/Data/delmas_famacha_data.json %d True 6" % (
            threshi, threshz, ndays), shell=True)
    print("***********************************************************************************************************")
    subprocess.call(
        "python create_dataset_heatmap.py F:/Data2/Heatmap F:/Data2/thresholded/delmas_70101200027_interpol_%d_zeros_%d_imputation_0 F:/Data2/gen_dataset_debug/delmas_70101200027_famachadays_%d_threshold_interpol_%d_threshold_zero2nan_%d --n_job=6" % (
            threshi, threshz, ndays, threshi, threshz), shell=True)
    print("***********************************************************************************************************")
    subprocess.call(
        "python ml_pipeline.py F:/Data/Pipeline/report_%d_%d_%d/ F:/Data2/gen_dataset_debug/delmas_70101200027_famachadays_%d_threshold_interpol_%d_threshold_zero2nan_%d/**/*.csv 2 10 0 0 0" % (
            threshi, threshz, ndays, ndays, threshi, threshz), shell=True)
