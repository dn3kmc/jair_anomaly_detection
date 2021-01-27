import sys  
sys.path.append("../time_series")  
sys.path.append("../anomaly_detection_methods")
from pbad_method import pbad_method
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
ts_name_list = []
score_number_list = []

window_size_list = []
window_incr_list = []
alphabet_size_list = []
relative_minsup_list = []
jaccard_threshold_list = []


mypath = "../jair_work_step_one_determine_characteristics/"
for f in listdir(mypath):
	for l in range(0,num_scores):
		if "ts_object" in f:
			ts = joblib.load(mypath + f)

			# parameter section	
			if ts.name == "international-airline-passengers_filled.csv":
				# parameter section
				window_size = 6
				window_incr = 3
			else:
				window_size = 12
				window_incr = 6

			alphabet_size = 100
			relative_minsup = .01
			jaccard_threshold = .9


			###
			ts_name_list.append(ts.name)
			ts_list.append(ts)
			score_number_list.append(l)

			window_size_list.append(window_size)
			window_incr_list.append(window_incr)
			alphabet_size_list.append(alphabet_size)
			relative_minsup_list.append(relative_minsup)
			jaccard_threshold_list.append(jaccard_threshold)


for q, (score_number, name, ts, window_size, window_incr, alphabet_size, relative_minsup, jaccard_threshold) in enumerate(zip(score_number_list, ts_name_list, ts_list, window_size_list, window_incr_list, alphabet_size_list, relative_minsup_list, jaccard_threshold_list)):
	print(score_number, name, ts, window_size, window_incr, alphabet_size, relative_minsup, jaccard_threshold)


all_scores = Parallel(n_jobs=num_cores)(delayed(pbad_method)(ts_obj=ts, window_size=window_size,
															 window_incr=window_incr, alphabet_size=alphabet_size,
															 relative_minsup=relative_minsup, jaccard_threshold=jaccard_threshold,
															 plot_anomaly_score=False) for (ts, window_size, window_incr, alphabet_size, relative_minsup, jaccard_threshold) in zip(ts_list,  window_size_list, window_incr_list, alphabet_size_list, relative_minsup_list, jaccard_threshold_list))

for q, (score_number, name, ts, window_size, window_incr, alphabet_size, relative_minsup, jaccard_threshold) in enumerate(zip(score_number_list, ts_name_list, ts_list, window_size_list, window_incr_list, alphabet_size_list, relative_minsup_list, jaccard_threshold_list)):
	print(score_number, name, ts, window_size, window_incr, alphabet_size, relative_minsup, jaccard_threshold)
	joblib.dump(all_scores[q], "pbad_scores_" + str(score_number) + "_" + name + "_window_size_" + str(window_size) + "_window_incr_" + str(window_incr) + "_alphabet_size_" + str(alphabet_size) + "_relative_minsup_" + str(relative_minsup) + "_jaccard_threshold_" + str(jaccard_threshold))











































