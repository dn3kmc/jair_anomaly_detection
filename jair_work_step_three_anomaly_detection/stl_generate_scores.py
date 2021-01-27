import sys  
sys.path.append("../time_series")  
sys.path.append("../anomaly_detection_methods")
import anomaly_detection_methods_helpers as ah
from stl_method import stl
from time_series import TimeSeries
from os import listdir
from os.path import isfile, join
import joblib
import pandas as pd

import multiprocessing
from joblib import Parallel, delayed



# number of times to generate scores
num_scores = 10
# 12 total cores
total_cores = multiprocessing.cpu_count()
num_cores = 5


############### first generate predictions 10 times for every data set using a gaussian window of 128 #################

gaussian_window_size = 128
step_size = int(gaussian_window_size/2)

ts_list = []
ts_name_list = []
score_number_list = []

swindow_list = []
sdegree_list = []
twindow_list = []
tdegree_list = []
inner_list = []
outer_list = []

stl_grid_search_df = pd.read_csv("../jair_work_step_two_grid_search/stl_grid_search_results.csv")

mypath = "../jair_work_step_one_determine_characteristics/"
for f in listdir(mypath):
	for l in range(0,num_scores):
		if "ts_object" in f:
			ts = joblib.load(mypath + f)


			if ts.get_period() >= 4:
				best_df = stl_grid_search_df.loc[[stl_grid_search_df.loc[stl_grid_search_df["TS Name"] == ts.name, 'RMSE'].idxmin()]]

				# print(ts.name)
				swindow = int(best_df["swindows"].values[0])
				sdegree = best_df["sdegrees"].values[0]
				twindow = best_df["twindows"].values[0]
				tdegree = best_df["tdegrees"].values[0]
				inner = best_df["inners"].values[0]
				outer = best_df["outers"].values[0]

				###
				ts_name_list.append(ts.name)
				ts_list.append(ts)
				score_number_list.append(l)

				swindow_list.append(swindow)
				sdegree_list.append(sdegree)
				twindow_list.append(twindow)
				tdegree_list.append(tdegree)
				inner_list.append(inner)
				outer_list.append(outer)

for q, (score_number, name, ts, swindow, sdegree, twindow, tdegree, inner, outer) in enumerate(zip(score_number_list, ts_name_list, ts_list, swindow_list, sdegree_list, twindow_list, tdegree_list, inner_list, outer_list)):
	print(score_number, name, ts, swindow, sdegree, twindow, tdegree, inner, outer)





all_scores_128 = Parallel(n_jobs=num_cores)(delayed(stl)(ts_obj=ts, gaussian_window_size=gaussian_window_size, step_size=step_size, 
														 swindow=swindow, sdegree=sdegree, twindow=twindow, tdegree=tdegree, inner=inner, outer=outer,
														 plot_anomaly_score=False, plot_components=False, grid_search_mode=False) 
														for (ts, swindow, sdegree, twindow, tdegree, inner, outer) in zip(ts_list, swindow_list, sdegree_list, twindow_list, tdegree_list, inner_list, outer_list))



for q, (score_number, name, ts, swindow, sdegree, twindow, tdegree, inner, outer) in enumerate(zip(score_number_list, ts_name_list, ts_list, swindow_list, sdegree_list, twindow_list, tdegree_list, inner_list, outer_list)):
	print(score_number, name, ts, swindow, sdegree, twindow, tdegree, inner, outer)
	joblib.dump(all_scores_128[q], "stl_scores_" + str(score_number) + "_gaussian_window_128_" + name + "_swindow_" + str(swindow) + "_sdegree_" + str(sdegree) + "_twindow_" + str(twindow) + "_tdegree_" + str(tdegree) +  "_inner_" + str(inner) + "_outer_" + str(outer))















# ################ second generate anomaly scores for every prediction set for every gaussian window size starting from 256 ##########
# gaussian window sizes
gaussian_window_sizes = [256,512,768,1024]
# step size is half of the gaussian window size

stl_grid_search_df = pd.read_csv("../jair_work_step_two_grid_search/stl_grid_search_results.csv")

mypath = "../jair_work_step_one_determine_characteristics/"
for f in listdir(mypath):
	for score_number in range(0,num_scores):
		for gaussian_window_size in gaussian_window_sizes:
			if "ts_object" in f:
				ts = joblib.load(mypath + f)
				if ts.get_period() >= 4:


					best_df = stl_grid_search_df.loc[[stl_grid_search_df.loc[stl_grid_search_df["TS Name"] == ts.name, 'RMSE'].idxmin()]]

					swindow = int(best_df["swindows"].values[0])
					sdegree = best_df["sdegrees"].values[0]
					twindow = best_df["twindows"].values[0]
					tdegree = best_df["tdegrees"].values[0]
					inner = best_df["inners"].values[0]
					outer = best_df["outers"].values[0]

					name = ts.name
					result_128 = joblib.load("stl_scores_" + str(score_number) + "_gaussian_window_128_" + name + "_swindow_" + str(swindow) + "_sdegree_" + str(sdegree) + "_twindow_" + str(twindow) + "_tdegree_" + str(tdegree) +  "_inner_" + str(inner) + "_outer_" + str(outer))

					stl_remainder = list(result_128["STL Remainder"])
					actual = list(ts.dataframe["value"].values)
					step_size = int(gaussian_window_size/2)
					anomaly_scores = ah.determine_anomaly_scores_error(stl_remainder, [0] * ts.get_length(), ts.get_length(), gaussian_window_size, step_size)

					result_dict = {"Anomaly Scores": anomaly_scores,
								   "Time": result_128["Time"],
								   "STL Remainder": stl_remainder}

					joblib.dump(result_dict, "stl_scores_" + str(score_number) + "_gaussian_window_" + str(gaussian_window_size) + "_" +  name + "_swindow_" + str(swindow) + "_sdegree_" + str(sdegree) + "_twindow_" + str(twindow) + "_tdegree_" + str(tdegree) +  "_inner_" + str(inner) + "_outer_" + str(outer))





