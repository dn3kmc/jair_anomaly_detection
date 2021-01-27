import pandas as pd
import time
import anomaly_detection_methods_helpers as ah
import matplotlib.pyplot as plt
import sys  
sys.path.append("../characteristics")
import characteristics_helpers as ch
import numpy as np
import copy
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error

def fbprophet(ts_obj, gaussian_window_size, step_size, changepoint_prior_scale=.05,
              growth='linear', yearly_seasonality='auto',
              weekly_seasonality='auto', daily_seasonality='auto',
              holidays=None, seasonality_mode='additive',
              seasonality_prior_scale=10, holidays_prior_scale=10,
              plot_anomaly_score=False, plot_forecast=False, grid_search_mode=False):


	start = time.time()

	fb_prophet_model = Prophet(changepoint_prior_scale=changepoint_prior_scale,
	                           growth=growth,
	                           yearly_seasonality=yearly_seasonality,
	                           weekly_seasonality=weekly_seasonality,
	                           daily_seasonality=daily_seasonality,
	                           holidays=holidays,
	                           seasonality_mode=seasonality_mode,
	                           seasonality_prior_scale=seasonality_prior_scale,
	                           holidays_prior_scale=holidays_prior_scale)
	if ts_obj.miss:
		# https://facebook.github.io/prophet/docs/outliers.html
		# Prophet has no problem with missing data
		# You set the missing values to NaNs in the training data
		# But you LEAVE the dates in the prediction
		ref_date_range = ch.get_ref_date_range(ts_obj.dataframe, ts_obj.dateformat, ts_obj.timestep)
		data_copy = copy.deepcopy(ts_obj.dataframe)
		data_copy["timestamp"] = pd.to_datetime(data_copy["timestamp"], format=ts_obj.dateformat)
		data_copy.set_index('timestamp', inplace=True)    
		data_copy = data_copy.reindex(ref_date_range, fill_value=np.nan)
		# use entire time series for training
		counts = [i for i in range(len(data_copy))]
		fb_df_train = pd.DataFrame({"count": counts,"ds": ref_date_range, "y": data_copy["value"]})
	else:
		# use entire time series for training
	    fb_df_train = pd.DataFrame({"ds": ts_obj.dataframe["timestamp"], "y": ts_obj.dataframe["value"]})

	fb_prophet_model.fit(fb_df_train, verbose=False)

	# periods=how much further you want to extend from the training dataset
	# this is not periodicity relating to seasonality
	future = fb_prophet_model.make_future_dataframe(periods=0, freq=ts_obj.timestep)
	# make a forecast over the entire time series
	fcst = fb_prophet_model.predict(future)

	predictions = fcst["yhat"].values

	# get RMSE
	if grid_search_mode:
		if ts_obj.miss:
			# remove the predictions from missing time steps
			inds = fb_df_train.loc[pd.isna(fb_df_train["y"]), :]["count"].values
			print(inds)
			nonmissing_predictions = []
			for i in range(len(predictions)):
				if i not in inds:
					nonmissing_predictions.append(predictions[i])
			rmse = mean_squared_error(ts_obj.dataframe["value"].values, nonmissing_predictions, squared=False)
			print("RMSE: ", rmse)

		else:
			rmse = mean_squared_error(ts_obj.dataframe["value"].values, predictions, squared=False)
			print("RMSE: ", rmse)
		return rmse

	# get anomaly scores
	else:

		if ts_obj.miss:
			# you HAVE to interpolate to get a gaussian window
			new_ts_obj = copy.deepcopy(ts_obj)
			new_ts_obj.set_miss(fill=True)
			actual = list(new_ts_obj.dataframe["value"])
		else:
		    actual = ts_obj.dataframe["value"]

		anomaly_scores = ah.determine_anomaly_scores_error(actual, predictions, ts_obj.get_length(), gaussian_window_size, step_size)

		end = time.time()

		if plot_forecast:
		    plt.plot([i for i in range(len(fcst))],fcst["yhat"])
		    plt.fill_between([i for i in range(len(fcst))], fcst["yhat_lower"], fcst["yhat_upper"], facecolor='blue', alpha=.3)
		    if ts_obj.miss:
		        plt.plot([i for i in range(len(predictions))], data_copy["value"], alpha=.5)
		    else:
		        plt.plot([i for i in range(len(predictions))], ts_obj.dataframe["value"], alpha=.5)
		    plt.xticks(rotation=90)
		    plt.show()

		if plot_anomaly_score:
			plt.subplot(211)
			plt.title("Anomaly Scores")
			plt.plot(anomaly_scores)
			plt.ylim([.99,1])
			plt.subplot(212)
			plt.title("Time Series")
			plt.plot(ts_obj.dataframe["value"].values)   
			plt.axvline(ts_obj.get_probationary_index(), color="black", label="probationary line")
			plt.tight_layout()
			plt.show()


		return {"Anomaly Scores": anomaly_scores,
		        "Time": end - start,
		        "Predictions": predictions}

