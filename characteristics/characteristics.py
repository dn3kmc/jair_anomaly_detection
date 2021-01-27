import numpy as np
import pandas as pd
import characteristics_helpers as ch


def has_miss(df, date_format, time_step_size, fill=True):
	ref_date_range = ch.get_ref_date_range(df, date_format, time_step_size)
	gaps = ref_date_range[~ref_date_range.isin(df["timestamp"])]
	if list(gaps):
		if fill:
			# print("Has missing time steps, but will fill.")
			filled_df = ch.fill_df(df, time_step_size, ref_date_range)
			return False, filled_df
		else:
			# print("Has missing time steps, but will NOT fill.")
			# STILL NEED TO REMOVE DUPLICATES!!!
			df = ch.remove_duplicate_time_steps(df["timestamp"][0], list(df["timestamp"])[-1], time_step_size, df, value="value")
			return True, df
	else:
		# print("No missing time steps.")
		return False, df

def how_many_miss(df, date_format, time_step_size):
	ref_date_range = ch.get_ref_date_range(df, date_format, time_step_size)
	gaps = ref_date_range[~ref_date_range.isin(df["timestamp"])]
	return len(gaps)

def has_concept_drift(df):
	vals = df["value"]
	unflattened_post_probs, chosen_vmax, epsilon, thresh_run = ch.concept_drift(vals, plot=False, verbose=True)
	run_length = 0
	for m in range(len(unflattened_post_probs)):
		if (unflattened_post_probs[m][m] < chosen_vmax) or (abs(unflattened_post_probs[m][m] - chosen_vmax) < epsilon):
			run_length += 1

		# diagonal is not cutting it for us
		else:
			for n in range(m):
				if (unflattened_post_probs[n][m] < chosen_vmax) or (abs(unflattened_post_probs[n][m] - chosen_vmax) < epsilon):
					if run_length >= thresh_run:
						return True

	return False


def has_seasonality(df):
	vals = df["value"]
	freq = ch.r_find_frequency(vals)
	if freq == 1:
		return False, 1  
	else:
		return True, freq


# def has_trend(df):
# 	vals = df["value"]
# 	if ch.has_stochastic_trend(vals) or ch.has_deterministic_trend(vals):
# 		return True
# 	else:
# 		return False


def has_trend(df):
	vals = df["value"]
	bool_val, type_trend = ch.trend_test(vals)
	return bool_val, type_trend




