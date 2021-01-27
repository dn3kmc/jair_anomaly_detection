import sys  
sys.path.append("../time_series")  
from time_series import TimeSeries
import time_series_helpers as th
import random
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time
import joblib

# # to pull from directory directly
# do_not_include = ["test_daily-minimum-temperatures-in-me.csv",
# 				  "test_art_noisy.csv",
# 				  "test_daily-total-female-births-in-cal.csv"]
# to_include = []
# directory = "../datasets"
# for filename in os.listdir(directory):
#     if filename.endswith(".csv"):
#     	if filename not in do_not_include:
#         	# print(filename)
#         	to_include.append(filename)
#     else:
#         continue


# already pulled from directory
	# name: time step size, date format, outlier type, numenta
to_include = {'Twitter_volume_FB_filled.csv':["5min","%Y-%m-%d %H:%M:%S", "P","Y"], 
			  'elb_request_count_8c0756_filled.csv':["5min","%Y-%m-%d %H:%M:%S","P,C","Y"], 
			  'FARM_filled.csv':["30min","%Y-%m-%d %H:%M:%S", "P,C","N"], 
			  'ec2_cpu_utilization_ac20cd_nofill.csv':["5min","%Y-%m-%d %H:%M:%S", "P,C","Y"], 
			  'ambient_temperature_system_failure_filled.csv':["1H","%Y-%m-%d %H:%M:%S", "P","Y"], 
			  'ibm-common-stock-closing-prices_nofill.csv':["1D","%Y-%m-%d", "C","N"], 
			  'rds_cpu_utilization_cc0c53_filled.csv':["5min","%Y-%m-%d %H:%M:%S","P,C","Y"], 
			  'exchange-2_cpc_results_nofill.csv':["1H","%Y-%m-%d %H:%M:%S", "C","Y"], 
			  'exchange-2_cpm_results_nofill.csv':["1H","%Y-%m-%d %H:%M:%S","P","Y"], 
			  'ec2_cpu_utilization_ac20cd_filled.csv':["5min","%Y-%m-%d %H:%M:%S","P,C","Y"], 
			  'rds_cpu_utilization_cc0c53_nofill.csv':["5min","%Y-%m-%d %H:%M:%S","P,C","Y"], 
			  'rds_cpu_utilization_e47b3b_filled.csv':["5min","%Y-%m-%d %H:%M:%S","P,C","Y"], 
			  'international-airline-passengers_filled.csv':["1MS","%Y-%m","P","N"], 
			  'art_daily_flatmiddle_filled.csv':["5min","%Y-%m-%d %H:%M:%S","C","Y"], 
			  'artificial_cd_3_filled.csv':["5min","%Y-%m-%d %H:%M:%S","P,C","N"], 
			  'grok_asg_anomaly_filled.csv':["5min","%Y-%m-%d %H:%M:%S", "P,C","Y"], 
			  'ec2_cpu_utilization_5f5533_filled.csv':["5min","%Y-%m-%d %H:%M:%S","P,C","Y"], 
			  'exchange-3_cpc_results_filled.csv':["1H","%Y-%m-%d %H:%M:%S","P","Y"], 
			  'exchange-2_cpm_results_filled.csv':["1H","%Y-%m-%d %H:%M:%S","P","Y"], 
			  'exchange-3_cpm_results_nofill.csv':["1H","%Y-%m-%d %H:%M:%S","P","Y"], 
			  'artificial_cd_1_filled.csv':["5min","%Y-%m-%d %H:%M:%S","P,C","N"],
			  'exchange-3_cpc_results_nofill.csv':["1H","%Y-%m-%d %H:%M:%S","P","Y"], 
			  'exchange-3_cpm_results_filled.csv':["1H","%Y-%m-%d %H:%M:%S","P","Y"], 
			  'artificial_cd_3_nofill.csv':["5min","%Y-%m-%d %H:%M:%S","P,C","N"], 
			  'artificial_cd_2_filled.csv':["5min","%Y-%m-%d %H:%M:%S","P,C","N"], 
			  'elb_request_count_8c0756_nofill.csv':["5min","%Y-%m-%d %H:%M:%S","P,C","Y"], 
			  'all_data_gift_certificates_filled.csv':["1H","%Y-%m-%d %H:%M:%S","P,C","N"], 
			  'Twitter_volume_AMZN_filled.csv':["5min","%Y-%m-%d %H:%M:%S","P,C","Y"], 
			  'exchange-2_cpc_results_filled.csv':["1H","%Y-%m-%d %H:%M:%S","C","Y"], 
			  'artificial_cd_1_nofill.csv':["5min","%Y-%m-%d %H:%M:%S","P,C","N"], 
			  'art_daily_nojump_filled.csv':["5min","%Y-%m-%d %H:%M:%S","C","Y"], 
			  'ibm-common-stock-closing-prices_filled.csv':["1D","%Y-%m-%d","C","N"], 
			  'ambient_temperature_system_failure_nofill.csv':["1H","%Y-%m-%d %H:%M:%S","P","Y"], 
			  'Twitter_volume_GOOG_filled.csv':["5min","%Y-%m-%d %H:%M:%S","P,C","Y"]}



NAMES = []
NUMBER_OF_TIME_STEPS = []
TIME_STEP_SIZES = []
MINIMUMS = []
MAXIMUMS = []
MEDIANS = []
MEANS = []
NUMBER_OF_ANOMALIES = []
OUTLIER_TYPES = []
NUMBER_OF_MISSING_TIME_STEPS = []
MISSING_TIMES = []
FROM_NUMENTA = []
SEASONALITIES = []
PERIODICITIES = []
SEASONALITY_TIMES = []
TRENDS = []
TREND_TYPES = []
TREND_TIMES = []
CONCEPT_DRIFT = []
CONCEPT_DRIFT_TIMES = []

for filename in to_include:
	print(filename)

	NAMES.append(filename)

	df = pd.read_csv('../datasets/' + filename, header=0)
	ts = TimeSeries(df, timestep=to_include[filename][0], dateformat=to_include[filename][1], name=filename)

	NUMBER_OF_TIME_STEPS.append(ts.get_length())

	TIME_STEP_SIZES.append(ts.get_timestep())

	MINIMUMS.append(ts.get_min())

	MAXIMUMS.append(ts.get_max())

	MEDIANS.append(ts.get_median())

	MEANS.append(ts.get_mean())

	NUMBER_OF_ANOMALIES.append(len(ts.dataframe[ts.dataframe["outlier"]==1]))

	# # use the below code to determine outlier type
	# ys = ts.dataframe[ts.dataframe["outlier"]==1]["value"].values
	# xs = list(ts.dataframe[ts.dataframe["outlier"]==1].index)
	# plt.plot(ts.dataframe["value"])
	# plt.scatter(xs,ys,marker="x",c="red",zorder=200)
	# plt.title(filename)
	# plt.show()

	OUTLIER_TYPES.append(to_include[filename][2])

	start = time.time()
	NUMBER_OF_MISSING_TIME_STEPS.append(ts.get_how_many_miss())
	ts.set_miss(fill=False)
	end = time.time()

	MISSING_TIMES.append(end-start)

	FROM_NUMENTA.append(to_include[filename][3])

	start = time.time()
	ts.set_seasonality()
	end = time.time()

	SEASONALITIES.append(ts.get_seasonality())

	PERIODICITIES.append(ts.get_period())

	SEASONALITY_TIMES.append(end-start)

	start = time.time()
	ts.set_trend()
	end = time.time()

	TRENDS.append(ts.get_trend())

	TREND_TYPES.append(ts.get_trend_type())

	TREND_TIMES.append(end-start)

	start = time.time()
	ts.set_concept_drift()
	end = time.time()

	CONCEPT_DRIFT.append(ts.get_concept_drift())

	CONCEPT_DRIFT_TIMES.append(end-start)

	# save this time series object
	joblib.dump(ts, filename[:-4] + "_ts_object")

	

table_df = pd.DataFrame({"Name": NAMES,
						 "# of time steps": NUMBER_OF_TIME_STEPS,
						 "Time Step Size": TIME_STEP_SIZES,
						 "Minimum": MINIMUMS,
						 "Maximum": MAXIMUMS,
						 "Median": MEDIANS,
						 "Mean": MEANS,
						 "# of Anomalies": NUMBER_OF_ANOMALIES,
						 "Outlier Type": OUTLIER_TYPES,
						 "# of Missing Time Steps": NUMBER_OF_MISSING_TIME_STEPS,
						 "Missing Time": MISSING_TIMES,
						 "From Numenta": FROM_NUMENTA,
						 "Seasonality": SEASONALITIES,
						 "Periodicity": PERIODICITIES,
						 "Seasonality Time": SEASONALITY_TIMES,
						 "Trend": TRENDS,
						 "Trend Type": TREND_TYPES,
						 "Trend Time": TREND_TIMES,
						 "Concept Drift": CONCEPT_DRIFT,
						 "Concept Drift Time": CONCEPT_DRIFT_TIMES})


joblib.dump(table_df, "table_df")