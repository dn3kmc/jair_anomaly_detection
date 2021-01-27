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

mypath = "../jair_work_step_one_determine_characteristics/"

growth = 'linear'
holidays = None
holidays_prior_scale = 10 
yearly_seasonality = 'auto'
weekly_seasonality = 'auto'
daily_seasonality = 'auto'
gaussian_window_size = 100 
step_size = int(gaussian_window_size/2) 

fb_grid_search_df = pd.read_csv("fb_grid_search_results.csv")
for f in listdir(mypath):
	if "ts_object" in f:
		ts = joblib.load(mypath + f)
		best_df = fb_grid_search_df.loc[[fb_grid_search_df.loc[fb_grid_search_df["TS Name"] == ts.name, 'RMSE'].idxmin()]]
		print(ts.name)
		seasonality_prior_scale = best_df["seasonality_prior_scales"].values[0]
		print(seasonality_prior_scale)
		changepoint_prior_scale = best_df["changepoint_prior_scales"].values[0]
		print(changepoint_prior_scale)
		seasonality_mode = best_df["seasonality_modes"].values[0]
		print(seasonality_mode)

		fbprophet(ts, gaussian_window_size=gaussian_window_size, step_size=step_size,
					  changepoint_prior_scale=changepoint_prior_scale,
		              growth=growth, yearly_seasonality=yearly_seasonality,
		              weekly_seasonality=weekly_seasonality, daily_seasonality=daily_seasonality,
		              holidays=holidays, seasonality_mode=seasonality_mode,
		              seasonality_prior_scale=seasonality_prior_scale, holidays_prior_scale=holidays_prior_scale,
		              plot_anomaly_score=True, plot_forecast=True, grid_search_mode=False)








