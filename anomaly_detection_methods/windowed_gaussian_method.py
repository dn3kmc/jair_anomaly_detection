import time
import anomaly_detection_methods_helpers as ah
import matplotlib.pyplot as plt

def windowed_gaussian(ts_obj, gaussian_window_size, step_size, plot_anomaly_score=False):
    if ts_obj.miss:
        raise ValueError("Missing time steps. Cannot use Windowed Gaussian.")
    start = time.time()

    anomaly_scores = ah.determine_anomaly_scores_error(ts_obj.dataframe["value"].values, [0] * ts_obj.get_length(), ts_obj.get_length(), gaussian_window_size, step_size=step_size)

    # why is the below here?
    # anomaly_scores = np.nan_to_num(anomaly_scores)

    end = time.time()

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
            "Time": end - start}