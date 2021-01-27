import sys  
sys.path.append("../time_series")  
sys.path.append("../anomaly_detection_methods")

import anomaly_detection_methods_helpers as ah
from stl_method import stl
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
stl_grid_search_df = pd.read_csv("../jair_work_step_two_grid_search/stl_grid_search_results.csv")

mypath = "../jair_work_step_one_determine_characteristics/"
for f in listdir(mypath):
	roc_auc_dict_list = []
	auc_list = []
	for score_number in range(0,num_scores):
		for gaussian_window_size in gaussian_window_sizes:
			step_size = int(gaussian_window_size/2)
			if "ts_object" in f:
				ts = joblib.load(mypath + f)

				if ts.get_period() >= 4:
					best_df = stl_grid_search_df.loc[[stl_grid_search_df.loc[stl_grid_search_df["TS Name"] == ts.name, 'RMSE'].idxmin()]]

					# print(ts.name)
					swindow = int(best_df["swindows"].values[0])
					sdegree = best_df["sdegrees"].values[0]
					twindow = best_df["twindows"].values[0]
					tdegree = best_df["tdegrees"].values[0]
					inner = best_df["inners"].values[0]
					outer = best_df["outers"].values[0]

					name = ts.name
					result_dict = joblib.load("../jair_work_step_three_anomaly_detection/stl_scores/stl_scores_" + str(score_number) + "_gaussian_window_" + str(gaussian_window_size) + "_" +  name + "_swindow_" + str(swindow) + "_sdegree_" + str(sdegree) + "_twindow_" + str(twindow) + "_tdegree_" + str(tdegree) +  "_inner_" + str(inner) + "_outer_" + str(outer))

					scores = [0 if math.isnan(x) else float(x) for x in result_dict["Anomaly Scores"]]
					y = [0 if math.isnan(x) else int(x) for x in ts.dataframe["outlier"].values]

					fpr, tpr, thresholds = roc_curve(y, scores, pos_label=1)

					roc_auc = auc(fpr, tpr)

					roc_auc_dict = {"TS Name": ts.name, "FPRS": fpr, "TPRS": tpr, "Thresholds": thresholds, "AUC": roc_auc, "Dict Name": "stl_roc_auc_" + str(score_number) + "_gaussian_window_" + str(gaussian_window_size) + "_" +  name + "_swindow_" + str(swindow) + "_sdegree_" + str(sdegree) + "_twindow_" + str(twindow) + "_tdegree_" + str(tdegree) +  "_inner_" + str(inner) + "_outer_" + str(outer)}
					roc_auc_dict_name = roc_auc_dict["Dict Name"]
					joblib.dump(roc_auc_dict, "stl_roc_auc_all/" + roc_auc_dict_name)
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

		# print(best_roc_auc_dict["Thresholds"])

		plt.figure()
		lw = 2
		plt.plot(fpr, tpr, color='darkorange',
		lw=lw, label='ROC (AUC = %0.2f)' % roc_auc)
		plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('STL ROC for ' + ts.name[:-4])
		plt.legend(loc="lower right")
		plt.tight_layout()
		plt.show()

		# save the one with the best auc
		joblib.dump(best_roc_auc_dict, "stl_roc_auc_best/" + roc_auc_dict_name)



