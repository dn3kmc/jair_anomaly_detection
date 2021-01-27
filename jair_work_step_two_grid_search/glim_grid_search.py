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

# grid search parameters
# see https://www.freecodecamp.org/news/how-to-pick-the-best-learning-rate-for-your-machine-learning-project-9c28865039a8/
etas_to_try = [.01, .1, 1, 10, 100]
# see page 87 in file:///home/cfreeman/Desktop/Tsagaris-T-2010-PhD-Thesis.pdf
lambdas_to_try = [.95, .96, .97, .98, .99, 1]



# to place in results csv
names = []
lambdas = []
etas = []
families = []
rmses = []
pass_fails = []
mypath = "../jair_work_step_one_determine_characteristics/"
for f in listdir(mypath):
	if "ts_object" in f:
		ts = joblib.load(mypath + f)
		# parameter section
		gaussian_window_size = 100 # does not matter for grid search mode as it is not used
		step_size = int(gaussian_window_size/2) # does not matter for grid search mode as it is not used
		# change the family to poisson if counting
		if "all_data_gift_certificates" in ts.name:
			family = "poisson"
		else:
			family ='gaussian'

		for eta in etas_to_try:
			for lambda_ in lambdas_to_try:
				names.append(ts.name)
				lambdas.append(lambda_)
				etas.append(eta)
				families.append(family)
				try:
					rmse = glim(ts, gaussian_window_size=gaussian_window_size, step_size=step_size, family=family, eta=eta, lambda_=lambda_, plot_anomaly_score=False, plot_forecast=False, grid_search_mode=True)
					rmses.append(rmse)
					pass_fails.append("Pass")
				except:
					rmses.append(np.inf)
					pass_fails.append("Fail")
					pass



glim_grid_search_df = pd.DataFrame({"TS Name": names, "Lambda": lambdas, "Eta": etas, "Family": families, "RMSE": rmses, "Pass":pass_fails})
glim_grid_search_df.to_csv("glim_grid_search_results.csv", index=False)


for f in listdir(mypath):
	if "ts_object" in f:
		ts = joblib.load(mypath + f)
		best_df = glim_grid_search_df.loc[[glim_grid_search_df.loc[glim_grid_search_df["TS Name"] == ts.name, 'RMSE'].idxmin()]]
		print(ts.name)
		print(best_df)







