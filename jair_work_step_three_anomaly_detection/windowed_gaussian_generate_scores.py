import sys  
sys.path.append("../time_series")  
sys.path.append("../anomaly_detection_methods")
import anomaly_detection_methods_helpers as ah
from windowed_gaussian_method import windowed_gaussian

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

ts_list = []
ts_name_list = []
score_number_list = []

gaussian_window_size_list = []
step_size_list = []

mypath = "../jair_work_step_one_determine_characteristics/"
for f in listdir(mypath):
	for l in range(0,num_scores):
		for g in [128, 256,512,768,1024]:
			if "ts_object" in f:
				ts = joblib.load(mypath + f)
				if not ts.miss:

					###
					ts_name_list.append(ts.name)
					ts_list.append(ts)
					score_number_list.append(l)

					gaussian_window_size_list.append(g)
					step_size_list.append(int(g/2))





for q, (score_number, name, ts, gaussian_window_size, step_size) in enumerate(zip(score_number_list, ts_name_list, ts_list, gaussian_window_size_list, step_size_list)):
	print(score_number, name, ts, gaussian_window_size, step_size)


all_scores = Parallel(n_jobs=num_cores)(delayed(windowed_gaussian)(ts_obj=ts, gaussian_window_size=gaussian_window_size, step_size=step_size, plot_anomaly_score=False) 
																for (ts, gaussian_window_size, step_size) in zip(ts_list, gaussian_window_size_list, step_size_list))



for q, (score_number, name, ts, gaussian_window_size, step_size) in enumerate(zip(score_number_list, ts_name_list, ts_list, gaussian_window_size_list, step_size_list)):
	print(score_number, name, ts, gaussian_window_size, step_size)
	joblib.dump(all_scores[q], "windowed_gaussian_scores_" + str(score_number) + "_gaussian_window_"+ str(gaussian_window_size) + "_" + name + "_step_size_" + str(step_size))














