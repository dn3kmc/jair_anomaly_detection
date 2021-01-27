import sys  
sys.path.append("../time_series") 
sys.path.append("../anomaly_detection_methods") 
from glim_method import glim
from time_series import TimeSeries
from os import listdir
from os.path import isfile, join
import joblib
import pandas as pd
import numpy as np

mypath = "../jair_work_step_one_determine_characteristics/"


gaussian_window_size = 100 
step_size = int(gaussian_window_size/2) 

glim_grid_search_df = pd.read_csv("glim_grid_search_results.csv")
for f in listdir(mypath):
	if "ts_object" in f:
		ts = joblib.load(mypath + f)
		best_df = glim_grid_search_df.loc[[glim_grid_search_df.loc[glim_grid_search_df["TS Name"] == ts.name, 'RMSE'].idxmin()]]
		print(ts.name)
		lambda_ = best_df["Lambda"].values[0]
		eta = best_df["Eta"].values[0]
		family = best_df["Family"].values[0]
		print(lambda_)
		print(eta)
		print(family)

		glim(ts, gaussian_window_size=gaussian_window_size, step_size=step_size, 
			family=family, eta=eta, lambda_=lambda_, plot_anomaly_score=True, 
			plot_forecast=True, grid_search_mode=False)






