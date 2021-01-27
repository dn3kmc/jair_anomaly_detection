import sys  
sys.path.append("../time_series")  
sys.path.append("../anomaly_detection_methods")

import anomaly_detection_methods_helpers as ah
from pbad_method import pbad_method
from time_series import TimeSeries
from os import listdir
from os.path import isfile, join
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import math
import matplotlib.pyplot as plt

num_scores = 10


gaussian_window_sizes = [128, 256,512,768,1024]


mypath = "../jair_work_step_one_determine_characteristics/"
for f in listdir(mypath):
	roc_auc_dict_list = []
	auc_list = []
	for score_number in range(0,num_scores):
		for gaussian_window_size in gaussian_window_sizes:
			if "ts_object" in f:
				ts = joblib.load(mypath + f)
				if not ts.miss:
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

					name = ts.name
					result_dict = joblib.load("../jair_work_step_three_anomaly_detection/pbad_scores/pbad_scores_" + str(score_number) + "_" + name + "_window_size_" + str(window_size) + "_window_incr_" + str(window_incr) + "_alphabet_size_" + str(alphabet_size) + "_relative_minsup_" + str(relative_minsup) + "_jaccard_threshold_" + str(jaccard_threshold))

					scores = [0 if math.isnan(x) else float(x) for x in result_dict["Anomaly Scores"]]
					y = [0 if math.isnan(x) else int(x) for x in ts.dataframe["outlier"].values]

					# pbad score length is not length of time series due to choice of window_size and window_incr
					adjusted_length_scores = []
					for i in range(window_size):
						adjusted_length_scores.append(scores[0])
					for j in range(1,len(scores)):
						for k in range(0,window_incr):
							adjusted_length_scores.append(scores[j])
					if len(adjusted_length_scores) < len(y):
						while len(adjusted_length_scores) < len(y):
							adjusted_length_scores.append(0)
					else:
						adjusted_length_scores = adjusted_length_scores[0:len(y)]

					fpr, tpr, thresholds = roc_curve(y, adjusted_length_scores, pos_label=1)

					roc_auc = auc(fpr, tpr)

					roc_auc_dict = {"TS Name": ts.name, "FPRS": fpr, "TPRS": tpr, "Thresholds": thresholds, "AUC": roc_auc, "Dict Name": "pbad_roc_auc_" + str(score_number) + "_" + name + "_window_size_" + str(window_size) + "_window_incr_" + str(window_incr) + "_alphabet_size_" + str(alphabet_size) + "_relative_minsup_" + str(relative_minsup) + "_jaccard_threshold_" + str(jaccard_threshold)}
					roc_auc_dict_name = roc_auc_dict["Dict Name"]
					joblib.dump(roc_auc_dict, "pbad_roc_auc_all/" + roc_auc_dict_name)
					roc_auc_dict_list.append(roc_auc_dict)
					auc_list.append(roc_auc)

	if roc_auc_dict_list:
		# print(auc_list)
		max_index = auc_list.index(max(auc_list))
		# print(max_index)
		best_roc_auc_dict = roc_auc_dict_list[max_index]

		fpr = best_roc_auc_dict["FPRS"]
		tpr = best_roc_auc_dict["TPRS"]
		roc_auc = best_roc_auc_dict["AUC"]
		roc_auc_dict_name = best_roc_auc_dict["Dict Name"]

		plt.figure()
		lw = 2
		plt.plot(fpr, tpr, color='darkorange',
		lw=lw, label='ROC (AUC = %0.2f)' % roc_auc)
		plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('PBAD ROC for ' + ts.name[:-4])
		plt.legend(loc="lower right")
		plt.tight_layout()
		plt.show()

		# save the one with the best auc
		joblib.dump(best_roc_auc_dict, "pbad_roc_auc_best/" + roc_auc_dict_name)



