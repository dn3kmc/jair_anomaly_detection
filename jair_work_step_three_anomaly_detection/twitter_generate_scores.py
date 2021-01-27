import sys  
sys.path.append("../time_series")  
sys.path.append("../anomaly_detection_methods")
from twitter_method import twitter
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

max_anoms_list = []
direction_list = []
alpha_list = []


mypath = "../jair_work_step_one_determine_characteristics/"
for f in listdir(mypath):
	for l in range(0,num_scores):
		if "ts_object" in f:
			ts = joblib.load(mypath + f)
			if not ts.miss:
				if ts.period > 1:
					# parameter section
					max_anoms = .02
					direction = "pos"
					alpha = .05

					###
					ts_name_list.append(ts.name)
					ts_list.append(ts)
					score_number_list.append(l)

					max_anoms_list.append(max_anoms)
					direction_list.append(direction)
					alpha_list.append(alpha)


for q, (score_number, name, ts, max_anoms, direction, alpha) in enumerate(zip(score_number_list, ts_name_list, ts_list, max_anoms_list, direction_list, alpha_list)):
	print(score_number, name, ts, max_anoms, direction, alpha)

all_scores = Parallel(n_jobs=num_cores)(delayed(twitter)(ts_obj=ts, max_anoms=max_anoms, direction=direction, alpha=alpha, plot_anomaly_score=False) for (ts, max_anoms, direction, alpha) in zip(ts_list, max_anoms_list, direction_list, alpha_list))

for q, (score_number, name, ts, max_anoms, direction, alpha) in enumerate(zip(score_number_list, ts_name_list, ts_list, max_anoms_list, direction_list, alpha_list)):
	print(score_number, name, ts, max_anoms, direction, alpha)
	joblib.dump(all_scores[q], "twitter_scores_" + str(score_number) + "_" + name + "_max_anoms_" + str(max_anoms) + "_direction_" + str(direction) + "_alpha_" + str(alpha))











































