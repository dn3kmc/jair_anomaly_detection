import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import as_strided
import rpy2
import rpy2.robjects.packages
from rpy2.robjects import r, numpy2ri, pandas2ri
import math

numpy2ri.activate()
pandas2ri.activate()

forecast = rpy2.robjects.packages.importr("forecast")

def convert_outlier_index(df_length, outlier_index_list_1s):
    """
    Input:

    df_length = length of dataframe

    list of predicted outlier indices
    e.g. [2,5]

    Output:

    list of whether or not an outlier
    e.g. [0,0,1,0,0,1,...] for the above example
    """
    outlier_index_list = np.zeros(df_length)
    # print("df_length:", df_length)
    # print("len: ", len(outlier_index_list_1s))
    # print("outlier index list 1s: ", outlier_index_list_1s)
    for outlier_index in outlier_index_list_1s:
        outlier_index_list[outlier_index] = 1
    return outlier_index_list

def as_sliding_window(ts, wl):
    assert len(ts.shape) == 1
    ss = ts.strides[0]
    n = ts.shape[0]
    return as_strided(ts, shape=[n-wl+1, wl], strides=[ss, ss])


def normal_probability(x, mean, std):
    # Given the normal distribution specified by the mean and standard deviation
    # args, return the probability of getting samples > x. This is the
    # Q-function: the tail probability of the normal distribution.
    if x < mean:
        # Gaussian is symmetrical around mean, so flip to get the tail probability
        xp = 2 * mean - x
        return normal_probability(xp, mean, std)
    # Calculate the Q function with the complementary error function, explained
    # here: http://www.gaussianwaves.com/2012/07/q-function-and-error-functions
    z = (x - mean) / std
    return 0.5 * math.erfc(z / math.sqrt(2))


def determine_anomaly_scores_error(actual, predictions, data_length, window_size, step_size):
    anomaly_scores = []
    window_data = []
    step_buffer = []
    mean = 0
    std = 1
    errors = []
    for i in range(data_length):
        anomaly_score = 0.0
        input_value = actual[i] - predictions[i]
        errors.append(input_value - mean)
        if len(window_data) > 0:
            anomaly_score = 1 - normal_probability(input_value, mean, std)
        if len(window_data) < window_size:
            window_data.append(input_value)
            mean = np.mean(window_data)
            std = np.std(window_data)
            if std == 0.0:
                std = .000001
        else:
            step_buffer.append(input_value)
            if len(step_buffer) == step_size:
                # slide window forward by step_size
                window_data = window_data[step_size:]
                window_data.extend(step_buffer)
                # reset step_buffer
                step_buffer = []
                mean = np.mean(window_data)
                std = np.std(window_data)
                if std == 0.0:
                    std = .000001
        anomaly_scores.append(anomaly_score)
    return anomaly_scores



def auto_arima(endog, exog=None, freq=None):
    if freq is None:
        freq = 1
    # endog_r = r.ts(pandas2ri.py2ri(endog), freq=freq)
    # if using more recent version of rpy2, py2ri was renamed to py2rpy
    # see reference: https://stackoverflow.com/questions/55990529/module-rpy2-robjects-pandas2ri-has-no-attribute-ri2py
    endog_r = r.ts(pandas2ri.py2rpy(endog), freq=freq)
    autoarima_args = {
        "seasonal": True,
        "stationary": False,
        "trace": True,
        "max.order":20,
        "max.p":20,
        "max.q":20,
        "max.P":20,
        "max.Q":20,
        "max.D":20,
        "max.d":20,
        "start.p":1,
        "start.q":1,
        "start.P":1,
        "start.Q":1
    }
    if exog is not None:
        # add noise to avoid rank-deficient error for exog
        scale = np.std(exog.values)
        z = scale*1e-4*np.random.randn(*exog.shape)
        exog_r = r.matrix(exog.values+z, nrow=exog.shape[0],
                          ncol=exog.shape[1],
                          dimnames=[[], exog.columns.tolist()])
        fit_r = forecast.auto_arima(y=endog_r, xreg=exog_r, **autoarima_args)
    else:
        fit_r = forecast.auto_arima(y=endog_r, **autoarima_args)
    fit_dict = dict(fit_r.items())
    # for proof of this order see last comment:
    # https://stats.stackexchange.com/questions/178577/how-to-read-p-d-and-q-of-auto-arima
    p, q, P, Q, s, d, D = list(fit_dict["arma"])
    return (p, d, q), (P, D, Q, s)


def get_exogenous(data, date_format):
    # used by sarimax and glm
    if date_format == "%Y-%m-%d %H:%M:%S":
        X = pd.get_dummies(pd.DataFrame({
            'day_of_week': data.index.dayofweek.astype('category'),
            'hour_of_day': data.index.hour.astype('category'),
            'Intercept': np.ones_like(data),
        }, index=data.index));
    # e.g. international-airline-passengers
    elif date_format == "%Y-%m":
        X = pd.get_dummies(pd.DataFrame({
                'month_of_year': data.index.month.astype('category'),
                'Intercept': np.ones_like(data),
            }, index=data.index));
    # e.g. ibm-common-stock-closing-prices
    elif date_format == "%Y-%m-%d":
        X = pd.get_dummies(pd.DataFrame({
            'month_of_year': data.index.month.astype('category'),
            'day_of_month': data.index.day.astype('category'),
            'Intercept': np.ones_like(data),
        }, index=data.index));
    else:
        raise ValueError("Need to create a new case for date_format")
    return X.astype('float64')


class PredictionResults:
    def __init__(self, predictions, errors, estimates, precisions):
        self.predictions = predictions
        self.errors = errors
        self.estimates = estimates
        self.mse = np.mean(np.square(errors))
        self.precisions = precisions


class Poisson:
    def mean(self, x):
        return np.exp(x)
    def inv_variance(self, x):
        return np.exp(-x)
    def inv_mean(self, x):
        return np.log(x)
    @property
    def default_mean(self):
        return 1.0
    @property
    def default_scale(self):
        return None


class Gaussian:
    def mean(self, x):
        return x
    def inv_variance(self, x):
        return 1.0  # XXX
    def inv_mean(self, x):
        return x
    @property
    def default_mean(self):
        return 0.0
    @property
    def default_scale(self):
        return 1.0


