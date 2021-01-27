import pandas as pd
import time
import anomaly_detection_methods_helpers as ah
import matplotlib.pyplot as plt
import sys  
sys.path.append("../characteristics") 
import characteristics_helpers as ch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

def glim(ts_obj, gaussian_window_size, step_size, family='gaussian', eta=1.0, lambda_=0.9999, plot_anomaly_score=False, plot_forecast=False, grid_search_mode=False):
    """
    Invokes GLiM.

    :param ts_obj
        TimeSeries object
    :param gaussian_window_size:
        Gaussian window size for creating anomaly scores
    :param step_size:
            Step size for creating anomaly scores
    :param plot_anomaly_scores:
        Plot anomaly scores if True
    :param plot_forecast:
        Plot predictions vs observations if True
    """
    start = time.time()
     # there are missing time steps. fill them with NaNs
    if ts_obj.miss:
        ref_date_range = ch.get_ref_date_range(ts_obj.dataframe, ts_obj.dateformat, ts_obj.timestep)
        gaps = ref_date_range[~ref_date_range.isin(ts_obj.dataframe["timestamp"])]
        filled_df = ch.fill_df(ts_obj.dataframe, ts_obj.timestep, ref_date_range, "fill_nan")
        endog = filled_df.set_index('timestamp')['value']
        exog = ah.get_exogenous(endog, ts_obj.get_dateformat())
    else:
        endog = ts_obj.dataframe.set_index('timestamp')['value']
        exog = ah.get_exogenous(endog, ts_obj.get_dateformat())
    
    # use entire time series for training
    full = ts_obj.get_length()
    initial_mean = endog.iloc[:full].mean()
    initial_stddev = endog.iloc[:full].std()
    results = online_glim(endog, exog, family=family, eta=eta, lambda_=lambda_, initial_loc=initial_mean, initial_scale=initial_stddev, save_precision=True)

    if ts_obj.miss:

        filled_results_predictions_values = []
        for item in results.predictions.values:
            if item != item:
                filled_results_predictions_values.append(0)
            else:
                filled_results_predictions_values.append(item)

        filled_results_errors_values = []
        for item in results.errors.values:
            if item != item:
                filled_results_errors_values.append(0)
            else:
                filled_results_errors_values.append(item)


        filled_df["results_predictions"] = filled_results_predictions_values
        filled_df["results_errors_values"] = filled_results_errors_values
        filled_df = filled_df.dropna()
        results_predictions = filled_df["results_predictions"].values
        results_errors_values = filled_df["results_errors_values"].values
    else:
        results_predictions = results.predictions.values
        results_errors_values = results.errors.values

    if grid_search_mode:

        if plot_forecast:
            plt.plot(list(results_predictions), label="Predictions", alpha=.7)
            plt.plot(list(ts_obj.dataframe["value"].values), label="Data", alpha=.5)
            plt.legend()
            plt.show()
        
        rmse = mean_squared_error(ts_obj.dataframe["value"].values, results_predictions, squared=False)
        print("RMSE: ", rmse)
        return rmse

    # print(len(results_errors_values))
    # print(full)
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    anomaly_scores  = ah.determine_anomaly_scores_error(
        results_errors_values, np.zeros_like(results_errors_values),
        full, gaussian_window_size, step_size)

    end = time.time()

    if plot_forecast:
        plt.plot(list(results_predictions), label="Predictions", alpha=.7)
        plt.plot(list(ts_obj.dataframe["value"].values), label="Data", alpha=.5)
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

    return {
        'Anomaly Scores': anomaly_scores,
        'Time': end-start,
        "Predictions": results_predictions
    }


def online_glim(endog, exog, lambda_=1.0, eta=1.0, family='gaussian',
               initial_loc=None, initial_scale=None, save_precision=False):
    """
    :param lambda_: Decay factor for "RLS"
    :param eta: learning rate (kludge, helps with stability in some cases)
    :param link: specify the distribution
    :param initial_loc: initial value for location parameter (affects stability)
    :param initial_scale: initial value for scale parameter
    :param save_precision: store the precision matrix at each time step
    """
    link = {
        'gaussian': ah.Gaussian,
        'poisson': ah.Poisson,
    }[family]()
    N = endog.shape[0]
    M = exog.shape[1]
    errors = np.zeros(N)
    predictions = np.zeros(N)
    estimates = np.zeros((N,M))
    precisions = np.zeros((N,M,M)) if save_precision else None
    lambda_inv = np.reciprocal(lambda_)
    P = np.eye(M)
    w = np.zeros(M)
    if 'Intercept' in exog.columns:
        intercept_index = exog.columns.get_loc('Intercept')
        w[intercept_index] = link.inv_mean(initial_loc)
    for t in tqdm(range(N)):
        u, d = exog.values[t], endog.values[t]
        # predict
        z = w.dot(u)
        y = link.mean(z)
        # update
        if not np.isnan(d):
            xi = d - y
            var_inv = link.inv_variance(z)
            pi = P.dot(u)
            k = pi / (lambda_ + u.dot(pi))
            w = w + eta*k*xi*var_inv
            P = lambda_inv*(P - np.outer(k, u).dot(P))
        else:
            xi = np.nan
        # save state and predictions
        errors[t] = xi
        predictions[t] = y
        estimates[t] = w
        if save_precision:
            precisions[t] = P
    predictions = pd.Series(predictions, index=endog.index)
    errors = pd.Series(errors, index=endog.index)
    estimates = pd.DataFrame(estimates, columns=exog.columns, index=endog.index)
    return ah.PredictionResults(predictions, errors, estimates, precisions)
