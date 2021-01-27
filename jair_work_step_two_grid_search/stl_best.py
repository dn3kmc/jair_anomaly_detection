import sys  
sys.path.append("../time_series") 
sys.path.append("../anomaly_detection_methods") 
from stl_method import stl
from time_series import TimeSeries
from os import listdir
from os.path import isfile, join
import joblib
import pandas as pd
import numpy as np

mypath = "../jair_work_step_one_determine_characteristics/"


gaussian_window_size = 100 
step_size = int(gaussian_window_size/2) 

stl_grid_search_df = pd.read_csv("stl_grid_search_results.csv")
for f in listdir(mypath):
	if "ts_object" in f:
		ts = joblib.load(mypath + f)
		if ts.get_period() >= 4:
			print(ts.name)
			best_df = stl_grid_search_df.loc[[stl_grid_search_df.loc[stl_grid_search_df["TS Name"] == ts.name, 'RMSE'].idxmin()]]

			swindow = int(best_df["swindows"].values[0])
			sdegree = best_df["sdegrees"].values[0]
			twindow = best_df["twindows"].values[0]
			tdegree = best_df["tdegrees"].values[0]
			inner = best_df["inners"].values[0]
			outer = best_df["outers"].values[0]

			print(swindow)
			print(sdegree)
			print(twindow)
			print(tdegree)
			print(inner)
			print(outer)

			stl(ts, gaussian_window_size=gaussian_window_size, step_size=step_size, swindow=swindow, 
				sdegree=sdegree, twindow=twindow, tdegree=tdegree, inner=inner, outer=outer, 
				grid_search_mode=False, plot_components=True, plot_anomaly_score=True)






