import joblib
import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy as np
import scipy.stats as st
import random
import sys  
sys.path.append("../time_series") 
from time_series import TimeSeries

from evaluation_methods import point_f_score, windowed_f_score, get_nab_score


anomaly_detection_methods = ["anomalous", "fb", "glim", "hstree", "htm", "matrix_profile", "pbad", "sarimax", "stl", "twitter", "vae", "windowed_gaussian"]
characteristics = ["seasonality", "trend", "conceptdrift", "missing"]
num_scores = 10

column_method_result_df = []
column_characteristic_result_df = []
column_evaluation_metric_result_df = []
column_evaluation_metric_score_result_df = []
column_evaluation_metric_upper_result_df = []
column_evaluation_metric_lower_result_df = []

# for the purposes of windowed f score friedman tests
df_windowed_f_score_dataset = []
df_windowed_f_score = []
df_num_score = []
df_method = []

for method_y in anomaly_detection_methods:
	for characteristic_x in characteristics:
		point_f_scores = []
		windowed_f_scores = []
		nab_scores = []
		auc_scores = []
		characteristic_path = "../datasets_" + characteristic_x + "/"
		for characteristic_dataset_z in listdir(characteristic_path):

			ts = joblib.load("../jair_work_step_one_determine_characteristics/" + characteristic_dataset_z[:-4] + "_ts_object")
			true_outliers = []
			for item in ts.dataframe["outlier"].values:
				if item != item:
					true_outliers.append(0)
				else:
					true_outliers.append(item)


			if method_y == "fb":
				roc_auc_best_path = "prophet_roc_auc_best/"
			else:
				roc_auc_best_path = method_y + "_roc_auc_best/"
			for roc_auc_best_dict in listdir(roc_auc_best_path):
				if characteristic_dataset_z in roc_auc_best_dict:
					auc_zy = joblib.load(roc_auc_best_path + roc_auc_best_dict)["AUC"]

					youden_path = "dataset_youden_thresholds/"

					if method_y == "anomalous":
						youden_threshold_method_string = "Anomalous"
					elif method_y == "fb":
						youden_threshold_method_string = "Prophet"
					elif method_y == "glim":
						youden_threshold_method_string = "GLiM"
					elif method_y == "hstree":
						youden_threshold_method_string = "HS Tree"
					elif method_y == "htm":
						youden_threshold_method_string = "HTM"
					elif method_y == "matrix_profile":
						youden_threshold_method_string = "Matrix Profile"
					elif method_y == "pbad":
						youden_threshold_method_string = "PBAD"
					elif method_y == "sarimax":
						youden_threshold_method_string = "SARIMAX"
					elif method_y == "stl":
						youden_threshold_method_string = "STL"
					elif method_y == "twitter":
						youden_threshold_method_string = "Twitter"
					elif method_y == "vae":
						youden_threshold_method_string = "VAE"
					elif method_y == "windowed_gaussian":
						youden_threshold_method_string = "Windowed Gaussian"
					else:
						raise ValueError("Key Error")


					threshold_zy = joblib.load(youden_path + characteristic_dataset_z + "_youden_thresholds")[youden_threshold_method_string]

					# print("Anomaly Detection Method: ", method_y)
					# print("Time Series Characteristic: ", characteristic_x)
					# print("Characteristic Dataset: ", characteristic_dataset_z) 
					# print("The Maximum AUC from the dataset and method: ", auc_zy)
					# print("Best Youden Threshold from ROC with max AUC for dataset using method: ", threshold_zy)
					# print("\n")

					if auc_zy != auc_zy:
						auc_scores.append(0)
					else:
						auc_scores.append(auc_zy)


					nab_score = get_nab_score(characteristic_dataset_z, method_y)
					if nab_score != nab_score:
						nab_scores.append(0)
					else:
						nab_scores.append(nab_score)


					mypath = "../jair_work_step_three_anomaly_detection/" + method_y + "_scores/"
					a = roc_auc_best_dict.split("roc_auc_",1)[1]
					b = a.split("_",1)[1]
					for i in range(num_scores):
						for f in listdir(mypath):
							if b in f:
								if "_" + str(i) + "_" in f:
									anomaly_scores = joblib.load("../jair_work_step_three_anomaly_detection/" + method_y + "_scores/" + f)["Anomaly Scores"]

									if method_y == "pbad":
										c = f.split("window_size_", 1)[1]
										d = c.split("_",1)[0]
										e = f.split("window_incr_", 1)[1]
										g = e.split("_",1)[0]
										window_incr = int(g)
										window_size = int(d)
										# pbad score length is not length of time series due to choice of window_size and window_incr
										adjusted_length_scores = []
										for i in range(window_size):
											adjusted_length_scores.append(anomaly_scores[0])
										for j in range(1,len(anomaly_scores)):
											for k in range(0,window_incr):
												adjusted_length_scores.append(anomaly_scores[j])
										if len(adjusted_length_scores) < len(true_outliers):
											while len(adjusted_length_scores) < len(true_outliers):
												adjusted_length_scores.append(0)
										else:
											adjusted_length_scores = adjusted_length_scores[0:len(true_outliers)]
										anomaly_scores = adjusted_length_scores


									point_f_score_number = point_f_score(threshold_zy, anomaly_scores, true_outliers)
									if point_f_score_number != point_f_score_number:
										point_f_scores.append(0)
									else:
										point_f_scores.append(point_f_score_number)

									windowed_f_score_number = windowed_f_score(threshold_zy, anomaly_scores, true_outliers)
									if windowed_f_score_number != windowed_f_score_number:
										windowed_f_scores.append(0)
									else:
										windowed_f_scores.append(windowed_f_score_number)

									# for the purposes of friedman tests
									df_windowed_f_score_dataset.append(characteristic_dataset_z)
									df_windowed_f_score.append(windowed_f_score_number)
									df_num_score.append(i)
									df_method.append(method_y)


		# for the purposes of friedman tests
		df_windowed_f_score_csv = pd.DataFrame({"Dataset": df_windowed_f_score_dataset, "Anomaly Detection Method": df_method, "Num Score": df_num_score, "Windowed F-Score": df_windowed_f_score})
		df_windowed_f_score_csv.to_csv("windowed_f_scores.csv")


		print("Overall AUC of " + method_y + " on " + characteristic_x + ": ", np.mean(auc_scores))
		column_method_result_df.append(method_y)
		column_characteristic_result_df.append(characteristic_x)
		column_evaluation_metric_result_df.append("AUC")
		column_evaluation_metric_score_result_df.append(np.mean(auc_scores))
		stdev = st.sem(auc_scores)
		if stdev == 0:
			stdev = .01
		lower, upper = st.t.interval(alpha=0.95, df=len(auc_scores)-1, loc=np.mean(auc_scores), scale=stdev) 
		column_evaluation_metric_upper_result_df.append(upper)
		column_evaluation_metric_lower_result_df.append(lower)

		print("Overall Point F-Score of " + method_y + " on " + characteristic_x + ": ", np.mean(point_f_scores))
		column_method_result_df.append(method_y)
		column_characteristic_result_df.append(characteristic_x)
		column_evaluation_metric_result_df.append("Point F-Score")
		column_evaluation_metric_score_result_df.append(np.mean(point_f_scores))
		stdev = st.sem(point_f_scores)
		if stdev == 0:
			stdev = .01
		lower, upper = st.t.interval(alpha=0.95, df=len(point_f_scores)-1, loc=np.mean(point_f_scores), scale=stdev) 
		column_evaluation_metric_upper_result_df.append(upper)
		column_evaluation_metric_lower_result_df.append(lower)

		print("Overall Windowed F-Score of " + method_y + " on " + characteristic_x + ": ", np.mean(windowed_f_scores))
		column_method_result_df.append(method_y)
		column_characteristic_result_df.append(characteristic_x)
		column_evaluation_metric_result_df.append("Windowed F-Score")
		column_evaluation_metric_score_result_df.append(np.mean(windowed_f_scores))
		stdev = st.sem(windowed_f_scores)
		if stdev == 0:
			stdev = .01
		lower, upper = st.t.interval(alpha=0.95, df=len(windowed_f_scores)-1, loc=np.mean(windowed_f_scores), scale=stdev) 
		column_evaluation_metric_upper_result_df.append(upper)
		column_evaluation_metric_lower_result_df.append(lower)

		print("Overall NAB of " + method_y + " on " + characteristic_x + ": ", np.mean(nab_scores))
		column_method_result_df.append(method_y)
		column_characteristic_result_df.append(characteristic_x)
		column_evaluation_metric_result_df.append("NAB")
		column_evaluation_metric_score_result_df.append(np.mean(nab_scores))
		stdev = st.sem(nab_scores)
		if stdev == 0:
			stdev = .01
		lower, upper = st.t.interval(alpha=0.95, df=len(nab_scores)-1, loc=np.mean(nab_scores), scale=stdev) 
		column_evaluation_metric_upper_result_df.append(upper)
		column_evaluation_metric_lower_result_df.append(lower)

	print("\n")

result_df = pd.DataFrame({"Anomaly Detection Method": column_method_result_df,
						  "Characteristic": column_characteristic_result_df,
						  "Evaluation Metric": column_evaluation_metric_result_df,
						  "Metric Score": column_evaluation_metric_score_result_df,
						  "Metric Score Upper": column_evaluation_metric_upper_result_df,
						  "Metric Score Lower": column_evaluation_metric_lower_result_df})


with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(result_df)


joblib.dump(result_df, "characteristic_roc_auc_plots/result_df")
joblib.dump(result_df, "characteristic_point_f_score_plots/result_df")
joblib.dump(result_df, "characteristic_windowed_f_score_plots/result_df")
joblib.dump(result_df, "characteristic_nab_score_plots/result_df")