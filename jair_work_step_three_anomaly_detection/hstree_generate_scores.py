import sys  
sys.path.append("../time_series")  
sys.path.append("../anomaly_detection_methods")
from streamingHS_method import hs_tree
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

n_trees_list = []
height_list = []
window_size_list = []


mypath = "../jair_work_step_one_determine_characteristics/"
for f in listdir(mypath):
	for l in range(0,num_scores):
		if "ts_object" in f:
			ts = joblib.load(mypath + f)

			# parameter section
			n_trees = 25
			height = 15
			window_size = 250

			###
			ts_name_list.append(ts.name)
			ts_list.append(ts)
			score_number_list.append(l)

			n_trees_list.append(n_trees)
			height_list.append(height)
			window_size_list.append(window_size)


for q, (score_number, name, ts, n_trees, height, window_size) in enumerate(zip(score_number_list, ts_name_list, ts_list, n_trees_list, height_list, window_size_list)):
	print(score_number, name, ts, n_trees, height, window_size)



all_scores = Parallel(n_jobs=num_cores)(delayed(hs_tree)(ts_obj=ts, n_trees=n_trees, height=height, window_size=window_size, plot_anomaly_score=False) for (ts, n_trees, height, window_size) in zip(ts_list, n_trees_list, height_list, window_size_list))

for q, (score_number, name, ts, n_trees, height, window_size) in enumerate(zip(score_number_list, ts_name_list, ts_list, n_trees_list, height_list, window_size_list)):
	print(score_number, name, ts, n_trees, height, window_size)
	joblib.dump(all_scores[q], "hstree_scores_" + str(score_number) + "_" + name + "_n_trees_" + str(n_trees) + "_height_" + str(height) + "_window_size_" + str(window_size))











































