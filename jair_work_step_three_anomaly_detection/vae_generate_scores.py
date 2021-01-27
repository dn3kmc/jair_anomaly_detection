import sys  
sys.path.append("../time_series")  
sys.path.append("../anomaly_detection_methods")
import anomaly_detection_methods_helpers as ah
from vae_method import vae_donut
from time_series import TimeSeries
from os import listdir
from os.path import isfile, join
import joblib
import pandas as pd
import numpy as np

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

window_size_list = []
mcmc_iteration_list = []
latent_dim_list = []


mypath = "../jair_work_step_one_determine_characteristics/"
for f in listdir(mypath):
	for l in range(0,num_scores):
		if "ts_object" in f:
			ts = joblib.load(mypath + f)


			window_size = min(120, gaussian_window_size, int(ts.get_length() / 4))
			mcmc_iteration = 10
			latent_dim = 5

			###
			ts_name_list.append(ts.name)
			ts_list.append(ts)
			score_number_list.append(l)

			window_size_list.append(window_size)
			mcmc_iteration_list.append(mcmc_iteration)
			latent_dim_list.append(latent_dim)




for q, (score_number, name, ts, window_size, mcmc_iteration, latent_dim) in enumerate(zip(score_number_list, ts_name_list, ts_list, window_size_list, mcmc_iteration_list, latent_dim_list)):
	print(score_number, name, ts, window_size, mcmc_iteration, latent_dim)


all_scores_128 = Parallel(n_jobs=num_cores)(delayed(vae_donut)(ts_obj=ts, gaussian_window_size=gaussian_window_size, step_size=step_size, 
															   window_size=window_size, mcmc_iteration=mcmc_iteration, latent_dim=latent_dim,
															   plot_reconstruction=False, plot_anomaly_score=False) 
																for (ts, window_size, mcmc_iteration, latent_dim) in zip(ts_list, window_size_list, mcmc_iteration_list, latent_dim_list))



for q, (score_number, name, ts, window_size, mcmc_iteration, latent_dim) in enumerate(zip(score_number_list, ts_name_list, ts_list, window_size_list, mcmc_iteration_list, latent_dim_list)):
	print(score_number, name, ts, window_size, mcmc_iteration, latent_dim)
	joblib.dump(all_scores_128[q], "vae_scores_" + str(score_number) + "_gaussian_window_128_" + name + "_window_size_" + str(window_size) + "_mcmc_iteration_" + str(mcmc_iteration) + "_latent_dim_" + str(latent_dim))















# ################ second generate anomaly scores for every prediction set for every gaussian window size starting from 256 ##########
# gaussian window sizes
gaussian_window_sizes = [256,512,768,1024]
# step size is half of the gaussian window size


mypath = "../jair_work_step_one_determine_characteristics/"
for f in listdir(mypath):
	for score_number in range(0,num_scores):
		for gaussian_window_size in gaussian_window_sizes:
			if "ts_object" in f:
				ts = joblib.load(mypath + f)

				window_size = min(120, gaussian_window_size, int(ts.get_length() / 4))
				mcmc_iteration = 10
				latent_dim = 5
				
				name = ts.name
				result_128 = joblib.load("vae_scores_" + str(score_number) + "_gaussian_window_128_" + name + "_window_size_" + str(window_size) + "_mcmc_iteration_" + str(mcmc_iteration) + "_latent_dim_" + str(latent_dim))

				reconstruction_probabilities = list(result_128["Reconstruction Probabilities"])
				step_size = int(gaussian_window_size/2)
				anomaly_scores = ah.determine_anomaly_scores_error(reconstruction_probabilities, np.zeros_like(reconstruction_probabilities), ts.get_length(), gaussian_window_size, step_size)

				result_dict = {"Anomaly Scores": anomaly_scores,
							   "Time": result_128["Time"],
							   "Reconstruction Probabilities": reconstruction_probabilities}

				joblib.dump(result_dict, "vae_scores_" + str(score_number) + "_gaussian_window_" + str(gaussian_window_size) + "_" + name + "_window_size_" + str(window_size) + "_mcmc_iteration_" + str(mcmc_iteration) + "_latent_dim_" + str(latent_dim))





