from statsmodels.tsa.stattools import adfuller
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects

from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
import logging
rpy2_logger.setLevel(logging.ERROR)   # will display errors, but not warnings


import bayesian_changepoint_detection.online_changepoint_detection as oncd
from functools import partial
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import copy
import pandas as pd

from arch.unitroot import PhillipsPerron, KPSS

# # for use with cox stuart test in R
# webr = importr('webr')
# cox_stuart_test = robjects.r('cox.stuart.test')

# for use with findfrequency in R
forecast = importr("forecast")
findfrequency = robjects.r('findfrequency')

def concept_drift(vals, plot=False, verbose=False):
	# https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.pcolor.html#matplotlib.pyplot.pcolor
	# https://stackoverflow.com/questions/42687454/pcolor-data-plot-in-python
	R1, maxes = oncd.online_changepoint_detection(vals, partial(oncd.constant_hazard, 250), oncd.StudentT(0.1, .01, 1, 0))
	sparsity = 1 # 5, increase this if time is too long
	unflattened_post_probs = -np.log(R1[0:-1:sparsity, 0:-1:sparsity])
	post_probs = (-np.log(R1[0:-1:sparsity, 0:-1:sparsity])).flatten()
	chosen_vmax = int(np.percentile(post_probs, 5))

	if plot:
		plt.pcolor(np.array(range(0, len(R1[:, 0]), sparsity)),
	               np.array(range(0, len(R1[:, 0]), sparsity)),
	               unflattened_post_probs,
	               cmap=cm.gray, vmin=0, vmax=chosen_vmax)
		plt.xlabel("time steps")
		plt.ylabel("run length")
		cbar = plt.colorbar(label="P(run)")
		cbar.set_ticks([0, chosen_vmax])
		cbar.set_ticklabels([1, 0])
		plt.show()

	epsilon = 2 * abs(unflattened_post_probs[1][1] - chosen_vmax) # a run consists of a diagonal of values that must be within this epsilon from chosen_vmax
	thresh_run = .1 * len(vals) # a run must be at least 10% the length of the time series before a concept drift can occur
	if verbose:
		print("vmax: ", chosen_vmax)
		print("epsilon: ", epsilon)
		print("threshold length for a run: ", thresh_run)

	return unflattened_post_probs, chosen_vmax, epsilon, thresh_run


def trend_test(vals):
	pp_result = PhillipsPerron(vals).pvalue
	kpss_result = KPSS(vals).pvalue

	# reject both. technically, we do not know and should consider different kinds of long range dependencies
	if (pp_result <= .05) and (kpss_result <= .05):
		return "Unknown", "Unknown"

	# reject H0 of pp, do not reject H0 of KPSS
	elif (pp_result <= .05) and (kpss_result > .05):
		return True, "deterministic"

	# reject H0 of KPSS. Do not reject H0 of pp
	elif (pp_result > .05) and (kpss_result <= .05):
		return True, "stochastic"

	# do not reject either. technically, we do not know and the time series is not informative enough
	else:
		return "Unknown", "Unknown"

# # uses adfuller test
# def has_stochastic_trend(vals):
# 	result = adfuller(vals)
# 	p_value = result[1]
# 	if p_value <= .05:
# 		return False
# 	else:
# 		return True


# # uses cox stuart test
# def has_deterministic_trend(vals):
# 	# https://stackoverflow.com/questions/25269655/how-to-pass-a-list-to-r-in-rpy2-and-get-result-back
# 	# http://rpy.sourceforge.net/rpy2/doc-2.1/html/robjects.html
# 	converted_vals = robjects.FloatVector(vals)
# 	result = cox_stuart_test(converted_vals)
# 	# print(result.names)
# 	# print(type(list(result)[1][0]))
# 	p_value = list(result)[1][0]
# 	if p_value <= .05:
# 		return True
# 	else:
# 		return False


def r_find_frequency(vals):
	converted_vals = robjects.FloatVector(vals)
	result = findfrequency(converted_vals)
	freq = result[0]
	return freq


def get_ref_date_range(df, date_format, time_step_size):
	df["timestamp"] = pd.to_datetime(df["timestamp"], format=date_format)
	df.index.name = None
	df = df.sort_values(by='timestamp')
	start_date = df["timestamp"].values[0]
	end_date = df["timestamp"].values[-1]
	# https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
	ref_date_range = pd.date_range(start_date, end_date, freq=time_step_size)
	return ref_date_range

def remove_duplicate_time_steps(start_date, end_date, time_step_size, given_dataframe, value="value"):
    """Remove duplicate time steps"""
    given_dataframe_copy = copy.deepcopy(given_dataframe)
    given_dataframe_copy.set_index('timestamp', inplace=True)
    # if duplicate records are found, keep only the first occurrence
    if len((given_dataframe_copy[given_dataframe_copy.index.duplicated()])) != 0:
        print("Duplicate records found: ")
        indexing = given_dataframe_copy[given_dataframe_copy.index.duplicated()].index[0]
        print(given_dataframe_copy.loc[given_dataframe_copy.index == indexing])
        print("Removing duplicates and keeping:  ")
        given_dataframe_copy = given_dataframe_copy[~given_dataframe_copy.index.duplicated()]
        print(given_dataframe_copy.loc[given_dataframe_copy.index == indexing])
    if "outlier" in given_dataframe.columns:
    	df = pd.DataFrame({"timestamp": given_dataframe_copy.index, value: given_dataframe_copy.value, "outlier": given_dataframe_copy["outlier"]})
    else:
    	df = pd.DataFrame({"timestamp": given_dataframe_copy.index, value: given_dataframe_copy.value})
    return df

def fill_df(df, time_step_size, ref_date_range, method="fill_value"):
	df = remove_duplicate_time_steps(df["timestamp"][0], list(df["timestamp"])[-1], time_step_size, df, value="value")
	data_copy = copy.deepcopy(df)
	data_copy.set_index('timestamp', inplace=True)    
	data_copy = data_copy.reindex(ref_date_range, fill_value=np.nan)
	if method == 'fill_nan':
		if "outlier" in df.columns:
			outlier_0_or_1 = []
			for item in data_copy["outlier"].values:
				if item != item:
					outlier_0_or_1.append(0)
				else:
					outlier_0_or_1.append(1)
			df = pd.DataFrame({"timestamp": ref_date_range, "value": data_copy["value"].values, 'outlier': outlier_0_or_1})
		else:
			df = pd.DataFrame({"timestamp": ref_date_range, "value": data_copy["value"].values})
		return df
	else:
		filled_data = data_copy.interpolate(method='linear', limit_direction='forward', axis=0)
		return filled_data