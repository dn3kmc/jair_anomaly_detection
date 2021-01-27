import sys  
sys.path.append("../time_series")  
sys.path.append("../anomaly_detection_methods")
from anomalous_method import anomalous
from time_series import TimeSeries
from os import listdir
from os.path import isfile, join
import joblib

import multiprocessing
from joblib import Parallel, delayed

# number of times to generate scores
num_scores = 10

# 12 total cores
total_cores = multiprocessing.cpu_count()
num_cores = 5

ts_list = []
ts_length_list = []
ts_name_list = []
score_number_list = []


mypath = "../jair_work_step_one_determine_characteristics/"
for f in listdir(mypath):
	for l in range(0,num_scores):
		if "ts_object" in f:
			ts = joblib.load(mypath + f)

			# parameter section
			if ts.get_length() > 1000:
				ts_length = 100
			else:
				ts_length = 25
			# see https://github.com/robjhyndman/anomalous-acm/issues/4
			if ts.name in ["art_daily_flatmiddle_filled.csv", "ambient_temperature_system_failure_nofill.csv"]:
				ts_length = 500

			###
			ts_name_list.append(ts.name)
			ts_list.append(ts)
			ts_length_list.append(ts_length)
			score_number_list.append(l)


# for q, (score_number, name, ts, ts_length) in enumerate(zip(score_number_list, ts_name_list, ts_list, ts_length_list)):
# 	print(score_number, name, ts, ts_length)

# for (ts, ts_length) in zip(ts_list, ts_length_list):
# 	print(ts, ts_length)


all_scores = Parallel(n_jobs=num_cores)(delayed(anomalous)(ts_obj=ts, ts_length=ts_length) for (ts, ts_length) in zip(ts_list, ts_length_list))

for q, (score_number, name, ts, ts_length) in enumerate(zip(score_number_list, ts_name_list, ts_list, ts_length_list)):
	print(q, score_number, name, ts, ts_length)
	joblib.dump(all_scores[q], "anomalous_scores_" + str(score_number) + "_" + name + "_ts_length_" + str(ts_length))











































