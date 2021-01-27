import time
import anomaly_detection_methods_helpers as ah
import matplotlib.pyplot as plt
import numpy as np
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects

# for use with Twitter AD in R
AnomalyDetection = importr("AnomalyDetection")
AnomalyDetectionVec= robjects.r('AnomalyDetectionVec')

# https://github.com/twitter/AnomalyDetection/blob/master/R/ts_anom_detection.R
def twitter(ts_obj, max_anoms, direction, alpha, plot_anomaly_score=False):
	"""
    Input:

    max_anoms =  Maximum number of anomalies that S-H-ESD will
    detect as a percentage of the data. Default is 10 percent

    direction = Directionality of the anomalies to be detected.
    Options are: 'pos', 'neg', 'both'. Default is 'pos'

    alpha = The level of statistical significance with which
    to accept or reject anomalies. Default is .05

    plot_anomalies = True or False. Whether or not to plot data with predicted anomalies

    Output:

    dictionary:
    anomaly scores, time
    """
	if ts_obj.period == 1:
		# this is because twitter uses R stl automatically
		# if there is no seasonality, seasonality decomposition is inappropriate
		# you will get this error: "series is not periodic or has less than two periods"
		# see line 162 in https://github.com/twitter/AnomalyDetection/blob/master/R/vec_anom_detection.R
		# technically you could turn it off: 
		# line 11 in https://github.com/twitter/AnomalyDetection/blob/master/R/detect_anoms.R
		# but this is a helper function and is automatically set to True in the main function AnomalyDetectionVec
		raise ValueError("Twitter AnomalyDetection can only handle seasonal datasets.")
	if ts_obj.miss:
		# proof line 35 in https://github.com/twitter/AnomalyDetection/blob/master/R/detect_anoms.R
		raise ValueError("Twitter AnomalyDetection cannot handle time series with missing time steps.")
	start = time.time()
	vals = list(ts_obj.dataframe["value"])
	converted_vals = robjects.FloatVector(vals)
	result = AnomalyDetectionVec(converted_vals, max_anoms=max_anoms, alpha=alpha, period=ts_obj.period, direction=direction, plot=robjects.vectors.BoolVector([True]))
	if len(result[0]) == 0:
		end = time.time()
		return {"Anomaly Scores": list(np.zeros(ts_obj.get_length())),
				"Time": end - start}
	anomaly_indices_r = list(result[0]["index"])
	# print(anomaly_indices_r)

	# indexing in Python starts at 0 while in R it starts at 1
	# 0 1 2 3 4 -> python
	# 1 2 3 4 5 -> R
	# so if R says index 5 is an anomaly, that is index 4 in python
	anomaly_indices_python = [i - 1  for i in anomaly_indices_r]

	# note that anomaly scores are just given the index
	# 0 if no anomaly
	# 1 if anomaly
	anomaly_scores = ah.convert_outlier_index(df_length=ts_obj.get_length(), outlier_index_list_1s=anomaly_indices_python)

	end = time.time()

	if plot_anomaly_score:
		plt.subplot(211)
		plt.title("Anomaly Scores")
		plt.plot(anomaly_scores)
		plt.subplot(212)
		plt.title("Time Series")
		plt.plot(ts_obj.dataframe["value"].values)   
		plt.axvline(ts_obj.get_probationary_index(), color="black", label="probationary line")
		plt.tight_layout()
		plt.show()

	return {"Anomaly Scores": anomaly_scores,
	        "Time": end - start}