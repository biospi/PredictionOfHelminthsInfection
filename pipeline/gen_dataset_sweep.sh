#!/bin/bash
# output_dir csv_db_dir_path famacha_file_path n_days_before_famacha resampling_resolution enable_graph_output n_process

#rm -rf /home/axel/dev_simulation/new_pipeline/thresholded/
njob=30
for threshz in `seq 360 120 720`;do
	for threshi in 10 15 20 30 45 60 80 100 120; do
		for reso in 1 3 5 10 15 20 25 30 45 60;do
        		for days in 1 2 3 4 5 6 7; do
                		python gen_training_sets.py /home/axel/dev_simulation/new_pipeline/dataset_build_start_z360/ /home/axel/dev_simulation/new_pipeline/thresholded/delmas_70101200027/interpol_${threshi}_zeros_${threshz}/ /home/axel/dev_simulation/data_south_africa/json/delmas_famacha_data.json $days ${reso}min False $njob
			done
		done
	done
done

