import sys  
sys.path.append("../time_series")  
sys.path.append("../anomaly_detection_methods")
import anomaly_detection_methods_helpers as ah
from matrixprofile_method import matrixprofile
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

############## first generate predictions 10 times for every data set using a gaussian window of 128 #################

gaussian_window_size = 128
step_size = int(gaussian_window_size/2)

ts_list = []
ts_name_list = []
score_number_list = []

subseq_len_list = []


mypath = "../jair_work_step_one_determine_characteristics/"
for f in listdir(mypath):
	for l in range(0,num_scores):
		if "ts_object" in f:
			ts = joblib.load(mypath + f)

			# print(ts.name)
			if ts.seasonality:
				subseq_len = ts.period
			else:
				subseq_len = 100
			if subseq_len < 5:
				subseq_len = 100

			###
			ts_name_list.append(ts.name)
			ts_list.append(ts)
			subseq_len_list.append(subseq_len)
			score_number_list.append(l)


for q, (score_number, name, ts, subseq_len) in enumerate(zip(score_number_list, ts_name_list, ts_list, subseq_len_list)):
	print(score_number, name, ts, subseq_len)


all_scores_128 = Parallel(n_jobs=num_cores)(delayed(matrixprofile)(ts_obj=ts, gaussian_window_size=gaussian_window_size, step_size=step_size, 
																subseq_len = subseq_len,
																plot_matrixprofile=False, plot_anomaly_score=False) 
																for (ts, subseq_len) in zip(ts_list, subseq_len_list))


for q, (score_number, name, ts, subseq_len) in enumerate(zip(score_number_list, ts_name_list, ts_list, subseq_len_list)):
	print(score_number, name, ts, subseq_len)
	joblib.dump(all_scores_128[q], "matrix_profile_scores_" + str(score_number) + "_gaussian_window_128_" + name + "_subseq_len_" + str(subseq_len))















################ second generate anomaly scores for every prediction set for every gaussian window size starting from 256 ##########
# gaussian window sizes
gaussian_window_sizes = [256,512,768,1024]
# step size is half of the gaussian window size


mypath = "../jair_work_step_one_determine_characteristics/"
for f in listdir(mypath):
	for score_number in range(0,num_scores):
		for gaussian_window_size in gaussian_window_sizes:
			if "ts_object" in f:
				ts = joblib.load(mypath + f)

				if ts.seasonality:
					subseq_len = ts.period
				else:
					subseq_len = 100
				if subseq_len < 5:
					subseq_len = 100
				name = ts.name
				result_128 = joblib.load("matrix_profile_scores_" + str(score_number) + "_gaussian_window_128_" + name + "_subseq_len_" + str(subseq_len))

				matrix_profile = list(result_128["Matrix Profile"])
				step_size = int(gaussian_window_size/2)

				anomaly_scores = ah.determine_anomaly_scores_error(matrix_profile, np.zeros_like(matrix_profile), ts.get_length(), gaussian_window_size, step_size)

				result_dict = {"Anomaly Scores": anomaly_scores,
							   "Time": result_128["Time"],
							   "Matrix Profile": matrix_profile}

				joblib.dump(result_dict, "matrix_profile_scores_" + str(score_number) + "_gaussian_window_" + str(gaussian_window_size) + "_" +  name + "_subseq_len_" + str(subseq_len))





