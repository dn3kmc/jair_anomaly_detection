import sys  
sys.path.append("../time_series")  
sys.path.append("../anomaly_detection_methods")
import anomaly_detection_methods_helpers as ah
from sarimax_method import sarimax
from time_series import TimeSeries
from os import listdir
from os.path import isfile, join
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
from joblib import Parallel, delayed

dataset_index = 10


all_datasets=['exchange-3_cpc_results_nofill.csv', # 0
			  'exchange-3_cpm_results_nofill.csv', # 1
			  'FARM_filled.csv', # 2**
			  'artificial_cd_1_nofill.csv', # 3
			  'Twitter_volume_GOOG_filled.csv', # 4
			  'exchange-2_cpc_results_nofill.csv', # 5
			  'grok_asg_anomaly_filled.csv', # 6
			  'ec2_cpu_utilization_5f5533_filled.csv', # 7
			  'elb_request_count_8c0756_filled.csv', # 8
			  'ec2_cpu_utilization_ac20cd_nofill.csv', # 9
			  'elb_request_count_8c0756_nofill.csv', # 10
			  'all_data_gift_certificates_filled.csv', # 11
			  'Twitter_volume_FB_filled.csv', #12
			  'art_daily_nojump_filled.csv', #13
			  'exchange-2_cpc_results_filled.csv', #14
			  'rds_cpu_utilization_cc0c53_nofill.csv', #15
			  'ibm-common-stock-closing-prices_filled.csv', #16
			  'ambient_temperature_system_failure_filled.csv', #17
			  'ec2_cpu_utilization_ac20cd_filled.csv', #18
			  'ambient_temperature_system_failure_nofill.csv', #19
			  'exchange-2_cpm_results_nofill.csv', #20
			  'artificial_cd_1_filled.csv', #21
			  'Twitter_volume_AMZN_filled.csv', #22
			  'artificial_cd_2_filled.csv', #23
			  'exchange-2_cpm_results_filled.csv', #24
			  'artificial_cd_3_nofill.csv', #25
			  'exchange-3_cpc_results_filled.csv', #26
			  'international-airline-passengers_filled.csv', #27
			  'rds_cpu_utilization_cc0c53_filled.csv', #28
			  'exchange-3_cpm_results_filled.csv', #29
			  'art_daily_flatmiddle_filled.csv', #30
			  'artificial_cd_3_filled.csv', #31
			  'rds_cpu_utilization_e47b3b_filled.csv', #32
			  'ibm-common-stock-closing-prices_nofill.csv'] #33





# number of times to generate scores
num_scores = 10
# 12 total cores
total_cores = multiprocessing.cpu_count()
if dataset_index == 2:
	num_cores = 1
else:
	num_cores = 5
# generally 5 is ok except for farm data set...in that case use 1



############### first generate predictions 10 times for every data set using a gaussian window of 128 #################

gaussian_window_size = 128
step_size = int(gaussian_window_size/2)

ts_list = []
ts_name_list = []
score_number_list = []


mypath = "../jair_work_step_one_determine_characteristics/"
for f in listdir(mypath):
	for l in range(0,num_scores):
		if "ts_object" in f:
			ts = joblib.load(mypath + f)
			if ts.name == all_datasets[dataset_index]: # bc sarimax is more prone to failure, we will do this per dataset

				###
				ts_name_list.append(ts.name)
				ts_list.append(ts)
				score_number_list.append(l)





for q, (score_number, name, ts) in enumerate(zip(score_number_list, ts_name_list, ts_list)):
	print(score_number, name, ts)


all_scores_128 = Parallel(n_jobs=num_cores)(delayed(sarimax)(ts_obj=ts, gaussian_window_size=gaussian_window_size, step_size=step_size, 
															 plot_forecast=False, plot_anomaly_score=False) 
															for ts in ts_list)



for q, (score_number, name, ts) in enumerate(zip(score_number_list, ts_name_list, ts_list)):
	print(score_number, name, ts)
	joblib.dump(all_scores_128[q], "sarimax_scores_" + str(score_number) + "_gaussian_window_128_" + name + "_step_size_" + str(step_size))















################ second generate anomaly scores for every prediction set for every gaussian window size starting from 256 ##########
# gaussian window sizes
gaussian_window_sizes = [256,512,768,1024]
# step size is half of the gaussian window size


mypath = "../jair_work_step_one_determine_characteristics/"
for f in listdir(mypath):
	for score_number in range(0,num_scores):
		for gaussian_window_size in gaussian_window_sizes:
			if "ts_object" in f:
				ts = joblib.load(mypath + f)
				if ts.name == all_datasets[dataset_index]:
					name = ts.name

					result_128 = joblib.load("sarimax_scores_" + str(score_number) + "_gaussian_window_128_" + name + "_step_size_" + str(64))

					forecast = list(result_128["Forecast"])
					actual = list(ts.dataframe["value"].values)

					step_size = int(gaussian_window_size/2)

					anomaly_scores = ah.determine_anomaly_scores_error(actual, forecast, len(forecast), gaussian_window_size, step_size)
					# plt.plot(anomaly_scores)
					# plt.show()


					result_dict = {"Anomaly Scores": anomaly_scores,
								   "Time": result_128["Time"],
								   "Predictions": forecast}

					joblib.dump(result_dict, "sarimax_scores_" + str(score_number) + "_gaussian_window_" + str(gaussian_window_size) + "_" + name + "_step_size_" + str(step_size))





