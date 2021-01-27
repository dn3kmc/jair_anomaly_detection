import time
import anomaly_detection_methods_helpers as ah
import matplotlib.pyplot as plt
import sys  
sys.path.append("../characteristics") 
import characteristics_helpers as ch
import numpy as np
from matrixprofile import *

def matrixprofile(ts_obj, subseq_len, gaussian_window_size, step_size, plot_matrixprofile=False, plot_anomaly_score=False):
    start = time.time()

    # see line 53
    # https://github.com/target/matrixprofile-ts/blob/master/matrixprofile/matrixProfile.py
    if ts_obj.miss:
        ref_date_range = ch.get_ref_date_range(ts_obj.dataframe, ts_obj.dateformat, ts_obj.timestep)
        gaps = ref_date_range[~ref_date_range.isin(ts_obj.dataframe["timestamp"])]
        filled_df = ch.fill_df(ts_obj.dataframe, ts_obj.timestep, ref_date_range, "fill_nan")
        # print("NaNs exist?: ",filled_df['value'].isnull().values.any())
        matrix_profile = matrixProfile.stamp(filled_df["value"].values,subseq_len)
    else:
        matrix_profile = matrixProfile.stamp(ts_obj.dataframe["value"].values,subseq_len)

    # Append np.nan to Matrix profile to enable plotting against raw data
    matrix_profile = np.append(matrix_profile[0],np.zeros(subseq_len-1)+np.nan)

    anomaly_scores = ah.determine_anomaly_scores_error(matrix_profile, np.zeros_like(matrix_profile), ts_obj.get_length(), gaussian_window_size, step_size)

    end = time.time()

    if plot_matrixprofile:
        plt.subplot(211)
        plt.title("Matrix Profile")
        plt.plot(matrix_profile)
        plt.subplot(212)
        plt.title("Time Series")
        plt.plot(ts_obj.dataframe["value"].values)   
        plt.axvline(ts_obj.get_probationary_index(), color="black", label="probationary line")
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

    return {
        'Anomaly Scores': anomaly_scores,
        'Time': end-start,
        "Matrix Profile": matrix_profile
    }


