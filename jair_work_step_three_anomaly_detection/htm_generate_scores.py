# convert csvs to htm score dictionaries
# check on lengths for aggregation
# check that all datasets are covered for every 0-10
import sys  
sys.path.append("../time_series")  
sys.path.append("../anomaly_detection_methods")
from htm_method import *
import anomaly_detection_methods_helpers as ah
from time_series import TimeSeries
from os import listdir
from os.path import isfile, join
import joblib
import pandas as pd

num_scores = 10



mypath = "../jair_work_step_one_determine_characteristics/"
for f in listdir(mypath):
	for l in range(0,num_scores):
		if "ts_object" in f:
			ts = joblib.load(mypath + f)
			if ts.get_length() >= 400:
				outer_path = "htm_scores/htm_output_csvs_" + str(l)
				result_dict = htm_method(ts, outer_path=outer_path, plot_anomaly_score=False)



				# check for aggregation errors (where you forget to turn off aggregation in htm studio)
				if len(result_dict["Anomaly Scores"]) != ts.get_length():
					print(l)
					print(ts.name)
					print(len(result_dict["Anomaly Scores"]))
					print(ts.get_length())

				# add _dict to the end or it will be a csv
				joblib.dump(result_dict, "htm_scores_" + str(l) + "_" + ts.name + "_dict")