import sys  
sys.path.append("../time_series") 
sys.path.append("../anomaly_detection_methods") 
from fbprophet_method import fbprophet
from time_series import TimeSeries
from os import listdir
from os.path import isfile, join
import joblib
import pandas as pd
import numpy as np

# grid search parameters
# see https://github.com/facebook/prophet/blob/master/python/fbprophet/forecaster.py
changepoint_prior_scales_to_try = [.01, .1, 1, 10, 100]
seasonality_prior_scales_to_try = [.01, .1, 1, 10, 100]



# these stay fixed
growth = 'linear'
holidays = None
holidays_prior_scale = 10 # this does not matter bc holidays is none
yearly_seasonality = 'auto'
weekly_seasonality = 'auto'
daily_seasonality = 'auto'
# to place in results csv
names = []
changepoint_prior_scales = []
seasonality_prior_scales = []
seasonality_modes = []
rmses = []
pass_fails = []
mypath = "../jair_work_step_one_determine_characteristics/"
for f in listdir(mypath):
	if "ts_object" in f:
		ts = joblib.load(mypath + f)
		# parameter section
		gaussian_window_size = 100 # does not matter for grid search mode as it is not used
		step_size = int(gaussian_window_size/2) # does not matter for grid search mode as it is not used

		for seasonality_prior_scale in seasonality_prior_scales_to_try:
			for changepoint_prior_scale in changepoint_prior_scales_to_try:
				names.append(ts.name)
				changepoint_prior_scales.append(changepoint_prior_scale)
				seasonality_prior_scales.append(seasonality_prior_scale)
				# https://kourentzes.com/forecasting/2014/11/09/additive-and-multiplicative-seasonality/
				if "international_airline" in ts.name:
					seasonality_mode = 'multiplicative'
				else:
					seasonality_mode = 'additive'
				seasonality_modes.append(seasonality_mode)
				try:
					rmse = fbprophet(ts, gaussian_window_size=gaussian_window_size, step_size=step_size,
					  changepoint_prior_scale=changepoint_prior_scale,
		              growth=growth, yearly_seasonality=yearly_seasonality,
		              weekly_seasonality=weekly_seasonality, daily_seasonality=daily_seasonality,
		              holidays=holidays, seasonality_mode=seasonality_mode,
		              seasonality_prior_scale=seasonality_prior_scale, holidays_prior_scale=holidays_prior_scale,
		              plot_anomaly_score=False, plot_forecast=False, grid_search_mode=True)
					rmses.append(rmse)
					pass_fails.append("Pass")
				except:
					rmses.append(np.inf)
					pass_fails.append("Fail")
					pass



fb_grid_search_df = pd.DataFrame({"TS Name": names, 
						 		  "seasonality_prior_scales": seasonality_prior_scales, 
						 		  "changepoint_prior_scales": changepoint_prior_scales, 
						 		  "seasonality_modes": seasonality_modes,
						 		  "RMSE": rmses, 
						 		  "Pass": pass_fails})

fb_grid_search_df.to_csv("fb_grid_search_results.csv", index=False)

for f in listdir(mypath):
	if "ts_object" in f:
		ts = joblib.load(mypath + f)
		best_df = fb_grid_search_df.loc[[fb_grid_search_df.loc[fb_grid_search_df["TS Name"] == ts.name, 'RMSE'].idxmin()]]
		print(ts.name)
		print(best_df)







