# import pandas as pd
import time
import anomaly_detection_methods_helpers as ah
import matplotlib.pyplot as plt
import sys   
sys.path.append("../characteristics") 
import characteristics_helpers as ch
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects

# for use with Anomalous in R

Anomalous = importr("anomalous")
tsmeasures= robjects.r('tsmeasures')
matrix = robjects.r('matrix')
ts_r = robjects.r('ts')
anomaly = robjects.r('anomaly')

def anomalous(ts_obj, ts_length, plot_anomaly_score=False):
    start = time.time()
    period = ts_obj.period
    # there are missing time steps. fill them with NaNs
    if ts_obj.miss:
        ref_date_range = ch.get_ref_date_range(ts_obj.dataframe, ts_obj.dateformat, ts_obj.timestep)
        gaps = ref_date_range[~ref_date_range.isin(ts_obj.dataframe["timestamp"])]
        filled_df = ch.fill_df(ts_obj.dataframe, ts_obj.timestep, ref_date_range, "fill_nan")
        ts = filled_df["value"].values
    else:
        ts = ts_obj.dataframe["value"].values

    # if time series is [1,2,3,4...] and ts_length is 3
    # this gives us [[1,2,3],[2,3,4]...]
    ts_strided = ah.as_sliding_window(ts, ts_length)

    # the number of mini time series
    num_ts = len(ts_strided)
    print("There are ", len(ts_strided), " many time series, each of length ", ts_length)

    # this gives us [1,2,3,2,3,4,3,4,5...]
    ts_strided = ts_strided.flatten()

    # create a matrix where every column is a mini time series
    a = matrix(ts_strided,ncol=num_ts)

    # make a become a time series object
    z = ts_r(a)

    # creates 13 features
    # y has shape: number of mini time series by 13
    # some features are nans if you have missing time steps
    try:
        y = tsmeasures(z) 
        # print(y.shape)

        # if there are missing values, they are IGNORED and DELETED using naomit
        # see line 9 in https://github.com/robjhyndman/anomalous/blob/master/R/detect_outliers.
        x = anomaly(y,plot=False)

        outlier_indices = list(x)[0]
    except:
        outlier_indices = []

    # the outlier indices are based on filled_df where missing time steps are filled with nans
    # we need to make them be based off of ts_obj.dataframe where missing time steps are not present
    if ts_obj.miss:
        anomaly_scores = ah.convert_outlier_index(len(filled_df), outlier_indices)
        # print(len(anomaly_scores))
        filled_df["anomaly scores"] = anomaly_scores
        # remove nans
        filled_df = filled_df.dropna()
        anomaly_scores = filled_df["anomaly scores"]
        # print(len(anomaly_scores))
        end = time.time()

    else:
        anomaly_scores = ah.convert_outlier_index(ts_obj.get_length(), outlier_indices)
        end = time.time()

    if plot_anomaly_score:
        plt.subplot(211)
        plt.title("Anomaly Scores")
        plt.plot(anomaly_scores)
        plt.subplot(212)
        plt.title("Time Series")
        plt.plot(ts_obj.dataframe["value"].values)   
        plt.axvline(ts_obj.get_probationary_index(), color="black", label="probationary line")
        plt.tight_layout()
        plt.show()

    return {"Anomaly Scores": anomaly_scores,
            "Time": end - start}