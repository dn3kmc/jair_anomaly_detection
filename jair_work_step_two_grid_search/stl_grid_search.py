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

# grid search parameters

# odd or string
swindows_to_try = [7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 'periodic']

# can only be 0,1,2
sdegrees_to_try = [0,1,2]

# odd
# Null would choose nextodd(ceiling((1.5*period) / (1-(1.5/s.window))))
# for a period of 12 that would be 23ish
twindows_to_try = [15, 25, 35, 45]

# can only be 0,1,2
tdegrees_to_try = [0,1,2]

# suggested to be 2 or fewer
inners_to_try = [1,2]

# Default is 0 but recommended if outliers are present
# https://www.rdocumentation.org/packages/stlplus/versions/0.5.1/topics/stlplus
outers_to_try = [0,1,5,10]


# to place in results csv
names = []
swindows = []
sdegrees = []
twindows = []
tdegrees = []
inners = []
outers = []
rmses = []
pass_fails = []
mypath = "../jair_work_step_one_determine_characteristics/"
for f in listdir(mypath):
	if "ts_object" in f:
		ts = joblib.load(mypath + f)
		if ts.get_period() >= 4:

			# parameter section
			gaussian_window_size = 100 # does not matter for grid search mode as it is not used
			step_size = int(gaussian_window_size/2) # does not matter for grid search mode as it is not used
			
			for swindow in swindows_to_try:
				for sdegree in sdegrees_to_try:
					for twindow in twindows_to_try:
						for tdegree in tdegrees_to_try:
							for inner in inners_to_try:
								for outer in outers_to_try:


									names.append(ts.name)
									swindows.append(swindow)
									sdegrees.append(sdegree)
									twindows.append(twindow)
									tdegrees.append(tdegree)
									inners.append(inner)
									outers.append(outer)


									try:
										rmse = stl(ts, gaussian_window_size=gaussian_window_size, step_size=step_size, swindow=swindow, 
												   sdegree=sdegree, twindow=twindow, tdegree=tdegree, inner=inner, outer=outer, 
												   grid_search_mode=True, plot_components=False, plot_anomaly_score=False)
										rmses.append(rmse)
										pass_fails.append("Pass")
									except:
										rmses.append(np.inf)
										pass_fails.append("Fail")
										pass


# technically it is sum of residuals and not rmse for stl but that's okay
stl_grid_search_df = pd.DataFrame({"TS Name": names, 
								   "swindows": swindows,
								   "sdegrees": sdegrees,
								   "twindows": twindows,
								   "tdegrees": tdegrees,
								   "inners": inners,
								   "outers": outers,
								   "RMSE": rmses, 
								   "Pass":pass_fails})

stl_grid_search_df.to_csv("stl_grid_search_results.csv", index=False)

for f in listdir(mypath):
	if "ts_object" in f:
		ts = joblib.load(mypath + f)
		if ts.get_period() >= 4:
			best_df = stl_grid_search_df.loc[[stl_grid_search_df.loc[stl_grid_search_df["TS Name"] == ts.name, 'RMSE'].idxmin()]]
			print(ts.name)
			print(best_df)







