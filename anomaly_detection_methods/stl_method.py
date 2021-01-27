import matplotlib.pyplot as plt
import sys  
sys.path.append("../characteristics")  
import characteristics_helpers as ch
import numpy as np
import time
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import anomaly_detection_methods_helpers as ah

# for use with stlplus in R
stlplus_library = importr("stlplus")
stlplus = robjects.r('stlplus')


def stl(ts_obj, gaussian_window_size, step_size, swindow, sdegree, twindow, tdegree, inner, outer, grid_search_mode=False, plot_components=False, plot_anomaly_score=False):
    # this method can deal with missing time steps
    if ts_obj.get_period() < 4:
        raise ValueError("n_periods must be at least 4.")
    start = time.time()

    if ts_obj.miss:
        ref_date_range = ch.get_ref_date_range(ts_obj.dataframe, ts_obj.get_dateformat(), ts_obj.get_timestep())
        nan_data = ch.fill_df(ts_obj.dataframe, ts_obj.get_timestep, ref_date_range, method="fill_nan")

        ts_values = nan_data["value"].values
        ts_timestamps = list(nan_data["timestamp"].values)
        ts_timestamps = np.array([str(item) for item in ts_timestamps])

        result = stlplus(ts_values, ts_timestamps, ts_obj.get_period(), swindow, sdegree, twindow, tdegree, inner, outer)
        stl_remainder = list(list(result)[0]["remainder"])

        nan_data["stl remainder"] = stl_remainder
        nan_data = nan_data.dropna()
        stl_remainder = nan_data["stl remainder"].values


    else:

        ts_values = ts_obj.dataframe["value"].values
        ts_timestamps = list(ts_obj.dataframe["timestamp"].values)
        ts_timestamps = np.array([str(item) for item in ts_timestamps])
        result = stlplus(ts_values, ts_timestamps, ts_obj.get_period(), swindow, sdegree, twindow, tdegree, inner, outer)
        stl_remainder = list(result)[0]["remainder"]
    
    if list(stl_remainder).count(0) >= int(.9*ts_obj.get_length()):
            raise ValueError("Remainders are mostly zero")

    if grid_search_mode:
        if plot_components:
            # print(list(result)[0]["remainder"].values)
            plt.subplot(311)
            plt.title("Seasonality")
            plt.plot(list(result)[0]["seasonal"].values)
            plt.subplot(312)
            plt.title("Trend")
            plt.plot(list(result)[0]["trend"].values)  
            plt.subplot(313) 
            plt.title("remainder")
            plt.plot(list(result)[0]["remainder"].values) 
            plt.show()
        the_sum = 0
        for remainder in stl_remainder:
            the_sum += abs(remainder)
        print("Sum of STL Remainders: ", the_sum)
        return the_sum

    anomaly_scores = ah.determine_anomaly_scores_error(stl_remainder, [0] * ts_obj.get_length(), ts_obj.get_length(), gaussian_window_size, step_size)


    end = time.time()

    if plot_components:
        # print(list(result)[0]["remainder"].values)
        plt.subplot(311)
        plt.title("Seasonality")
        plt.plot(list(result)[0]["seasonal"].values)
        plt.subplot(312)
        plt.title("Trend")
        plt.plot(list(result)[0]["trend"].values)  
        plt.subplot(313) 
        plt.title("remainder")
        plt.plot(list(result)[0]["remainder"].values)
        plt.tight_layout()
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
            "STL Remainder": stl_remainder}
    
