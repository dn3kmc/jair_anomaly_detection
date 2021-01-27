import time
import matplotlib.pyplot as plt
import sys  
sys.path.append("../pbad/src")
from methods.PreProcessor import PreProcessor
from methods.PBAD import PBAD




def pbad_method(ts_obj, window_size, window_incr, alphabet_size, relative_minsup, jaccard_threshold, plot_anomaly_score=False):

	start = time.time()
	if ts_obj.miss:
		raise ValueError("PBAD cannot handle missing time steps.")

	ts = {0:ts_obj.dataframe["value"].values}

	# pbad cannot take missing time steps. i tried it below:
	'''
	if ts_obj.miss:
		# raise ValueError("PBAD cannot handle missing time steps.")
		ref_date_range = ch.get_ref_date_range(ts_obj.dataframe, ts_obj.dateformat, ts_obj.timestep)
		gaps = ref_date_range[~ref_date_range.isin(ts_obj.dataframe["timestamp"])]
		filled_df = ch.fill_df(ts_obj.dataframe, ts_obj.timestep, ref_date_range, "fill_nan")
		print("NaNs exist?: ",filled_df['value'].isnull().values.any())
		ts = {0:filled_df["value"].values}
	else:
		ts = {0:ts_obj.dataframe["value"].values}
	'''

	# preprocess the data
	# in the supplement (see table 4 caption:"The window length and window increment used to divide 
	# each time series in fixed-size sliding windows, chosen in function of the sample rate ∆t of
	# the underlying dataset.") the general pattern is 
	# if time step size is 5 minutes, window_size = 1 hour
	# if time step size is 1 hour, window_size = 12 hour
	# if time step size is 30 minutes, window_size = 6 hour
	# in all cases above this means window_size=12, window_incr=6
	# no advice for time step sizes of 1D or 1MS, but we will use the same rules
	# window_incr is less than or equal to window_size. in 3 cases, it is half. we will use half
	# alphabet_size= nr. bins is between 30-100 in the supplement. we will use max

	preprocesser = PreProcessor(window_size=window_size, window_incr=window_incr, alphabet_size=alphabet_size, add_scaling=False)
	ts_windows_discretized, ts_windows, _, window_labels = preprocesser.preprocess(continuous_series=ts,return_undiscretized=True)

	pbad = PBAD(relative_minsup=relative_minsup, jaccard_threshold=jaccard_threshold, pattern_type='all', pattern_pruning='maximal')
	anomaly_scores = pbad.fit_predict(ts_windows, ts_windows_discretized) # pbad score length is not length of time series due to choice of window_size and window_incr

	end = time.time()

	if plot_anomaly_score:

		adjusted_length_scores = []
		for i in range(window_size):
			adjusted_length_scores.append(anomaly_scores[0])
		for j in range(1,len(anomaly_scores)):
			for k in range(0,window_incr):
				adjusted_length_scores.append(anomaly_scores[j])
		if len(adjusted_length_scores) < ts_obj.get_length():
			while len(adjusted_length_scores) < ts_obj.get_length():
				adjusted_length_scores.append(0)
		else:
			adjusted_length_scores = adjusted_length_scores[0:ts_obj.get_length()]

		plt.subplot(311)
		plt.title("Anomaly Scores")
		plt.plot(anomaly_scores)
		plt.subplot(312)
		plt.title("Adjusted Length Anomaly Scores")
		plt.plot(adjusted_length_scores)
		plt.subplot(313)
		plt.title("Time Series")
		plt.plot(ts_obj.dataframe["value"].values)
		plt.axvline(ts_obj.get_probationary_index(), color="black", label="probationary line")
		plt.tight_layout()
		plt.show()

	return {
	'Anomaly Scores': anomaly_scores,
	'Time': end-start,}













