import pandas as pd
import statsmodels.api as sm
import time
import anomaly_detection_methods_helpers as ah
import matplotlib.pyplot as plt
import sys  
sys.path.append("../characteristics") 
sys.path.append("../time_series")  
from time_series import TimeSeries
import characteristics_helpers as ch
import numpy as np
import pmdarima as pm
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

def sarimax(ts_obj, gaussian_window_size, step_size, plot_anomaly_score=False, plot_forecast=False):
    slide_size = 200
    if ts_obj.get_length() >= slide_size:
        n = slide_size
        list_df = [ts_obj.dataframe[i:i+n] for i in range(0,ts_obj.dataframe.shape[0],n)]

        anomaly_scores_list = []
        times_list = []
        forecasts_list = []
        for chunk_df in tqdm(list_df):
            print(ts_obj.name)
            if len(chunk_df) >= slide_size:
                chunk_ts_obj = TimeSeries(chunk_df, timestep=ts_obj.timestep, dateformat=ts_obj.dateformat, name=ts_obj.name)
                # NEED TO SET CHARACTERISTIC OF SEASONALITY ONLY
                chunk_ts_obj.set_seasonality()
                chunk_result = sarimax_mini(chunk_ts_obj, gaussian_window_size, step_size, plot_anomaly_score=False, plot_forecast=False)
                anomaly_scores_list.append(chunk_result["Anomaly Scores"])
                times_list.append(chunk_result["Time"])
                forecasts_list.append(chunk_result["Forecast"])

        anomaly_scores = []
        for sublist in anomaly_scores_list:
            for item in sublist:
                anomaly_scores.append(item)

        forecast = []
        for sublist in forecasts_list:
            for item in sublist:
                forecast.append(item)

        while len(anomaly_scores) < ts_obj.get_length():
            anomaly_scores.append(0)

        while len(forecast) < ts_obj.get_length():
            forecast.append(0)

        if plot_forecast:
            plt.plot(forecast, alpha=.7, label="Predictions")
            plt.plot(ts_obj.dataframe["value"].values, alpha=.5, label="Data")
            plt.legend()
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
        
        return {"Anomaly Scores": np.asarray(anomaly_scores),
                "Time": sum(times_list),
                "Forecast": forecast}

    else:
        return sarimax_mini(ts_obj, gaussian_window_size, step_size, plot_anomaly_score=plot_anomaly_score, plot_forecast=plot_forecast)

def sarimax_mini(ts_obj, gaussian_window_size, step_size, plot_anomaly_score=False, plot_forecast=False):

        start = time.time()

        # SARIMAX SHOULD BE ABLE TO HANDLE MISSING VALUES theoretically and code wise:
        # --theoretically: https://stats.stackexchange.com/questions/346225/fitting-arima-to-time-series-with-missing-values
        # "ARIMA models are state space models and the Kalman filter, 
        # which is used to fit state space models, deals with missing values exactly 
        # by simply skipping the update phase."
        # --code wise: https://www.statsmodels.org/devel/examples/notebooks/generated/statespace_sarimax_internet.html
        # "The novel feature is the ability of the model to work on datasets with missing values."

        # Unfortunately, python pyramid autoarima cannot handle missing values
        # I tested this using
        '''
        import pmdarima as pm
        from pmdarima import model_selection
        import numpy as np
        data = pm.datasets.load_wineind()
        data[50] = np.nan
        data[160] = np.nan
        train, test = model_selection.train_test_split(data, train_size=150)
        arima = pm.auto_arima(train, error_action='ignore', trace=True,
            suppress_warnings=True, maxiter=10,seasonal=True, m=12)
        '''
        # which resulted in ValueError: Input contains NaN, infinity or a value too large for dtype('float64').

        # R's autoarima can handle missing values BUT
        # you can force a seasonality with python pyramid with a specified periodicity
        # you cannot force a seasonality in R autoarima
        # I have tried upping parameters like in:
        # https://stackoverflow.com/questions/24390859/why-does-auto-arima-drop-my-seasonality-component-when-stepwise-false-and-approx
        # but it does not work

        # I will instead fill in missing values using interpolation and use python pyramid and FIX s
        # this will give me p,d,q,P,D,Q
        # then I will fill in missing values with NaNs and then use Statsmodels' fit

        if ts_obj.miss:
            ref_date_range = ch.get_ref_date_range(ts_obj.dataframe, ts_obj.dateformat, ts_obj.timestep)
            gaps = ref_date_range[~ref_date_range.isin(ts_obj.dataframe["timestamp"])]

            filled_df_value = ch.fill_df(ts_obj.dataframe, ts_obj.timestep, ref_date_range, "fill_value")
            filled_df_value = pd.DataFrame({"timestamp":filled_df_value.index,"value":filled_df_value["value"]})
            endogenous_values_filled_interpolate = filled_df_value.set_index('timestamp')['value']

            filled_df_nan = ch.fill_df(ts_obj.dataframe, ts_obj.timestep, ref_date_range, "fill_nan")
            endogenous_values_filled_nan = filled_df_nan.set_index('timestamp')['value']

            exogenous_values_filled_interpolate = ah.get_exogenous(endogenous_values_filled_interpolate, ts_obj.get_dateformat()).drop("Intercept", axis=1, errors='ignore')
            exogenous_values_filled_nan = ah.get_exogenous(endogenous_values_filled_nan, ts_obj.get_dateformat()).drop("Intercept", axis=1, errors='ignore')

            # fill NaNs with values using interpolation to use Pyramid
            try:
                arima = pm.auto_arima(y=endogenous_values_filled_interpolate, 
                                     exogenous=exogenous_values_filled_interpolate,
                                     max_p=3,
                                     max_q=3,
                                     error_action='ignore', 
                                     trace=True,
                                     suppress_warnings=True, 
                                     maxiter=1,
                                     maxorder=5,
                                     seasonal=ts_obj.seasonality, 
                                     m=ts_obj.period)
            # http://alkaline-ml.com/pmdarima/seasonal-differencing-issues.html
            except ValueError:
                 arima = pm.auto_arima(y=endogenous_values_filled_interpolate, 
                                     exogenous=exogenous_values_filled_interpolate,
                                     D=0,
                                     max_p=3,
                                     max_q=3,
                                     error_action='ignore', 
                                     trace=True,
                                     suppress_warnings=True, 
                                     maxiter=1,
                                     maxorder=5,
                                     seasonal=ts_obj.seasonality, 
                                     m=ts_obj.period)

            order = arima.order
            seasonal_order = arima.seasonal_order

            # order = (2,1,2)
            # seasonal_order = (0,0,1,3)

            # print(order)
            # print(seasonal_order)

            # use NaNs with sarimax fitting with statsmodels
            try:
                fit_result = sm.tsa.SARIMAX(endogenous_values_filled_nan, exogenous_values_filled_nan, order=order,
                                     seasonal_order=seasonal_order,
                                     time_varying=True, mle_regression=False).fit()
            # https://github.com/statsmodels/statsmodels/issues/5459
            # https://github.com/statsmodels/statsmodels/issues/5374
            except np.linalg.LinAlgError as err:
                print("\n\n!!!!")
                print("enforce_stationarity = False")
                fit_result = sm.tsa.SARIMAX(endogenous_values_filled_nan, exogenous_values_filled_nan, order=order,
                                     seasonal_order=seasonal_order,
                                     time_varying=True, mle_regression=False, enforce_stationarity=False).fit()


            model = sm.tsa.SARIMAX(endogenous_values_filled_nan, exogenous_values_filled_nan, order=order, seasonal_order=seasonal_order,
                             time_varying=True, mle_regression=False)
            model.initialize_known(fit_result.filtered_state[..., -1],
                                   fit_result.filtered_state_cov[..., -1])

            model.update(model.start_params)

            filter_result = model.ssm.filter()
            response = filter_result.forecasts.squeeze(0)

            # print(len(ts_obj.dataframe["value"]))
            # print(len(response))

            filled_df_nan["response"] = response
            filled_df_nan = filled_df_nan.dropna()
            response = filled_df_nan["response"].values

            # print(len(ts_obj.dataframe["value"]))
            # print(len(response))

            anomaly_scores = ah.determine_anomaly_scores_error(ts_obj.dataframe["value"], response, len(response), gaussian_window_size, step_size)
        else:
           endogenous_values = ts_obj.dataframe.set_index('timestamp')['value']
           exogenous_values = ah.get_exogenous(endogenous_values, ts_obj.get_dateformat()).drop("Intercept", axis=1, errors='ignore')

        try:
           arima = pm.auto_arima(y=endogenous_values, 
                                 exogenous=exogenous_values,
                                 max_p=3,
                                 max_q=3,
                                 error_action='ignore', 
                                 trace=True,
                                 suppress_warnings=True, 
                                 maxiter=1,
                                 maxorder=5,
                                 seasonal=ts_obj.seasonality, 
                                 m=ts_obj.period)
        # http://alkaline-ml.com/pmdarima/seasonal-differencing-issues.html
        except ValueError:
            arima = pm.auto_arima(y=endogenous_values, 
                                 exogenous=exogenous_values,
                                 D=0,
                                 max_p=3,
                                 max_q=3,
                                 error_action='ignore', 
                                 trace=True,
                                 suppress_warnings=True, 
                                 maxiter=1,
                                 maxorder=5,
                                 seasonal=ts_obj.seasonality, 
                                 m=ts_obj.period)


        order = arima.order
        seasonal_order = arima.seasonal_order

        # print("!!!!!!!!!!!!!!!")
        # print(ts_obj.name)           
        # print(order)
        # print(seasonal_order)
        # print("!!!!!!!!!!!!!!!")

        try:
           fit_result = sm.tsa.SARIMAX(endogenous_values, exogenous_values, order=order,
                     seasonal_order=seasonal_order,
                     time_varying=True, mle_regression=False).fit()
           model = sm.tsa.SARIMAX(endogenous_values, exogenous_values, order=order, seasonal_order=seasonal_order,
             time_varying=True, mle_regression=False)
        except np.linalg.LinAlgError as err:
            print("\n\n!!!!")
            print("enforce_stationarity = False")
            fit_result = sm.tsa.SARIMAX(endogenous_values, exogenous_values, order=order,
                     seasonal_order=seasonal_order,
                     time_varying=True, mle_regression=False, enforce_stationarity=False, simple_differencing=True).fit()
            model = sm.tsa.SARIMAX(endogenous_values, exogenous_values, order=order, seasonal_order=seasonal_order,
                 time_varying=True, mle_regression=False,enforce_stationarity=False, simple_differencing=True)

        model.initialize_known(fit_result.filtered_state[..., -1],
                   fit_result.filtered_state_cov[..., -1])
        model.update(model.start_params)
        filter_result = model.ssm.filter()
        response = filter_result.forecasts.squeeze(0)

        anomaly_scores = ah.determine_anomaly_scores_error(endogenous_values, response, len(response), gaussian_window_size, step_size)

        end = time.time()

        if plot_forecast:
            plt.plot(response, alpha=.7, label="Predictions")
            plt.plot(ts_obj.dataframe["value"].values, alpha=.5, label="Data")
            plt.legend()
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

        return {"Anomaly Scores": np.asarray(anomaly_scores),
                "Time": end - start,
                "Forecast": response}



