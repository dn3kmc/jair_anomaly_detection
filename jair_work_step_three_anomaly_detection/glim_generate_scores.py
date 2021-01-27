import sys  
sys.path.append("../time_series")  
sys.path.append("../anomaly_detection_methods")
import anomaly_detection_methods_helpers as ah
from glim_method import glim
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



############## first generate predictions 10 times for every data set using a gaussian window of 128 #################

gaussian_window_size = 128
step_size = int(gaussian_window_size/2)

ts_list = []
ts_name_list = []
score_number_list = []

lambdas = []
etas = []
families = []

glim_grid_search_df = pd.read_csv("../jair_work_step_two_grid_search/glim_grid_search_results.csv")
mypath = "../jair_work_step_one_determine_characteristics/"
for f in listdir(mypath):
	for l in range(0,num_scores):
		if "ts_object" in f:
			ts = joblib.load(mypath + f)

			best_df = glim_grid_search_df.loc[[glim_grid_search_df.loc[glim_grid_search_df["TS Name"] == ts.name, 'RMSE'].idxmin()]]
			# print(ts.name)
			lambda_ = best_df["Lambda"].values[0]
			eta = best_df["Eta"].values[0]
			family = best_df["Family"].values[0]

			###
			ts_name_list.append(ts.name)
			ts_list.append(ts)
			score_number_list.append(l)

			lambdas.append(lambda_)
			etas.append(eta)
			families.append(family)


for q, (score_number, name, ts, lambda_, eta, family) in enumerate(zip(score_number_list, ts_name_list, ts_list, lambdas, etas, families)):
	print(score_number, name, ts, lambda_, eta, family)


all_scores_128 = Parallel(n_jobs=num_cores)(delayed(glim)(ts_obj=ts, gaussian_window_size=gaussian_window_size, step_size=step_size,
														  family=family, eta=eta, lambda_=lambda_,
														  plot_anomaly_score=False, plot_forecast=False, grid_search_mode=False) 
														  for (ts, lambda_, eta, family) in zip(ts_list, lambdas, etas, families))



for q, (score_number, name, ts, lambda_, eta, family) in enumerate(zip(score_number_list, ts_name_list, ts_list, lambdas, etas, families)):
	print(score_number, name, ts, lambda_, eta, family)
	joblib.dump(all_scores_128[q], "glim_scores_" + str(score_number) + "_gaussian_window_128_" + name + "_lambda_" + str(lambda_) + "_eta_" + str(eta) + "_family_" + family)















# ################ second generate anomaly scores for every prediction set for every gaussian window size starting from 256 ##########
# gaussian window sizes
gaussian_window_sizes = [256,512,768,1024]
# step size is half of the gaussian window size

glim_grid_search_df = pd.read_csv("../jair_work_step_two_grid_search/glim_grid_search_results.csv")

mypath = "../jair_work_step_one_determine_characteristics/"
for f in listdir(mypath):
	for score_number in range(0,num_scores):
		for gaussian_window_size in gaussian_window_sizes:
			if "ts_object" in f:
				ts = joblib.load(mypath + f)

				best_df = glim_grid_search_df.loc[[glim_grid_search_df.loc[glim_grid_search_df["TS Name"] == ts.name, 'RMSE'].idxmin()]]
				lambda_ = best_df["Lambda"].values[0]
				eta = best_df["Eta"].values[0]
				family = best_df["Family"].values[0]
				name = ts.name

				result_128 = joblib.load("glim_scores_" + str(score_number) + "_gaussian_window_128_" + name + "_lambda_" + str(lambda_) + "_eta_" + str(eta) + "_family_" + family)

				predictions = list(result_128["Predictions"])
				actual = list(ts.dataframe["value"].values)
				step_size = int(gaussian_window_size/2)
				anomaly_scores = ah.determine_anomaly_scores_error(actual, predictions, ts.get_length(), gaussian_window_size, step_size)

				result_dict = {"Anomaly Scores": anomaly_scores,
							   "Time": result_128["Time"],
							   "Predictions": predictions}

				joblib.dump(result_dict, "glim_scores_" + str(score_number) + "_gaussian_window_" + str(gaussian_window_size) + "_" + name + "_lambda_" + str(lambda_) + "_eta_" + str(eta) + "_family_" + family)





