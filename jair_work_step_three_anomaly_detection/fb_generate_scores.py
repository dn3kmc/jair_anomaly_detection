import sys  
sys.path.append("../time_series")  
sys.path.append("../anomaly_detection_methods")
import anomaly_detection_methods_helpers as ah
from fbprophet_method import fbprophet
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

seasonality_prior_scale_list = []
changepoint_prior_scale_list = []
seasonality_mode_list = []

fb_grid_search_df = pd.read_csv("../jair_work_step_two_grid_search/fb_grid_search_results.csv")
mypath = "../jair_work_step_one_determine_characteristics/"
for f in listdir(mypath):
	for l in range(0,num_scores):
		if "ts_object" in f:
			ts = joblib.load(mypath + f)

			best_df = fb_grid_search_df.loc[[fb_grid_search_df.loc[fb_grid_search_df["TS Name"] == ts.name, 'RMSE'].idxmin()]]
			# print(ts.name)
			seasonality_prior_scale = best_df["seasonality_prior_scales"].values[0]
			changepoint_prior_scale = best_df["changepoint_prior_scales"].values[0]
			seasonality_mode = best_df["seasonality_modes"].values[0]

			###
			ts_name_list.append(ts.name)
			ts_list.append(ts)
			seasonality_prior_scale_list.append(seasonality_prior_scale)
			changepoint_prior_scale_list.append(changepoint_prior_scale)
			seasonality_mode_list.append(seasonality_mode)
			score_number_list.append(l)


for q, (score_number, name, ts, seasonality_prior_scale, changepoint_prior_scale, seasonality_mode) in enumerate(zip(score_number_list, ts_name_list, ts_list, seasonality_prior_scale_list, changepoint_prior_scale_list, seasonality_mode_list)):
	print(score_number, name, ts, seasonality_prior_scale, changepoint_prior_scale, seasonality_mode)


all_scores_128 = Parallel(n_jobs=num_cores)(delayed(fbprophet)(ts_obj=ts, gaussian_window_size=gaussian_window_size, step_size=step_size, 
																changepoint_prior_scale=changepoint_prior_scale,
																growth='linear', yearly_seasonality='auto',
																weekly_seasonality='auto', daily_seasonality='auto',
																holidays=None, seasonality_mode=seasonality_mode,
																seasonality_prior_scale=seasonality_prior_scale, holidays_prior_scale=10,
																plot_anomaly_score=False, plot_forecast=False, grid_search_mode=False) 
																for (ts, seasonality_prior_scale, changepoint_prior_scale, seasonality_mode) in zip(ts_list, seasonality_prior_scale_list, changepoint_prior_scale_list, seasonality_mode_list))



for q, (score_number, name, ts, seasonality_prior_scale, changepoint_prior_scale, seasonality_mode) in enumerate(zip(score_number_list, ts_name_list, ts_list, seasonality_prior_scale_list, changepoint_prior_scale_list, seasonality_mode_list)):
	print(score_number, name, ts, seasonality_prior_scale, changepoint_prior_scale, seasonality_mode)
	joblib.dump(all_scores_128[q], "fbprophet_scores_" + str(score_number) + "_gaussian_window_128_" + name + "_seasonality_prior_scale_" + str(seasonality_prior_scale) + "_changepoint_prior_scale_" + str(changepoint_prior_scale) + "_seasonality_mode_" + seasonality_mode )















# ################ second generate anomaly scores for every prediction set for every gaussian window size starting from 256 ##########
# gaussian window sizes
gaussian_window_sizes = [256,512,768,1024]
# step size is half of the gaussian window size

fb_grid_search_df = pd.read_csv("../jair_work_step_two_grid_search/fb_grid_search_results.csv")

mypath = "../jair_work_step_one_determine_characteristics/"
for f in listdir(mypath):
	for score_number in range(0,num_scores):
		for gaussian_window_size in gaussian_window_sizes:
			if "ts_object" in f:
				ts = joblib.load(mypath + f)

				best_df = fb_grid_search_df.loc[[fb_grid_search_df.loc[fb_grid_search_df["TS Name"] == ts.name, 'RMSE'].idxmin()]]
				seasonality_prior_scale = best_df["seasonality_prior_scales"].values[0]
				changepoint_prior_scale = best_df["changepoint_prior_scales"].values[0]
				seasonality_mode = best_df["seasonality_modes"].values[0]
				name = ts.name
				result_128 = joblib.load("fbprophet_scores_" + str(score_number) + "_gaussian_window_128_" + name + "_seasonality_prior_scale_" + str(seasonality_prior_scale) + "_changepoint_prior_scale_" + str(changepoint_prior_scale) + "_seasonality_mode_" + seasonality_mode)

				predictions = list(result_128["Predictions"])
				actual = list(ts.dataframe["value"].values)
				step_size = int(gaussian_window_size/2)
				anomaly_scores = ah.determine_anomaly_scores_error(actual, predictions, ts.get_length(), gaussian_window_size, step_size)

				result_dict = {"Anomaly Scores": anomaly_scores,
							   "Time": result_128["Time"],
							   "Predictions": predictions}

				joblib.dump(result_dict, "fbprophet_scores_" + str(score_number) + "_gaussian_window_" + str(gaussian_window_size) + "_" +  name + "_seasonality_prior_scale_" + str(seasonality_prior_scale) + "_changepoint_prior_scale_" + str(changepoint_prior_scale) + "_seasonality_mode_" + seasonality_mode )





