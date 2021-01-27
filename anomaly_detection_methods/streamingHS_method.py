import time
import matplotlib.pyplot as plt
import sys  
sys.path.append("../characteristics") 
import characteristics_helpers as ch
from creme import anomaly
from tqdm import tqdm


def hs_tree(ts_obj, n_trees, height, window_size, plot_anomaly_score=False):
    start = time.time()
    # there are missing time steps. fill them with NaNs
    if ts_obj.miss:
        # print("MISS")
        ref_date_range = ch.get_ref_date_range(ts_obj.dataframe, ts_obj.dateformat, ts_obj.timestep)
        gaps = ref_date_range[~ref_date_range.isin(ts_obj.dataframe["timestamp"])]
        filled_df = ch.fill_df(ts_obj.dataframe, ts_obj.timestep, ref_date_range, "fill_nan")
        ts = filled_df["value"].values
    else:
        ts = ts_obj.dataframe["value"].values

    # normalize (nans are possibly still present)
    the_max = max(ts)
    ts = [float(i)/the_max for i in ts]

    hst = anomaly.HalfSpaceTrees(n_trees=n_trees, height=height, window_size=window_size)

    for x in ts[:window_size]:
        hst = hst.fit_one({'x': x})

    anomaly_scores = []

    for x in tqdm(ts):
        features = {"x": x}
        hst = hst.fit_one(features)
        anomaly_scores.append(hst.score_one(features))
        # if math.isnan(x):
        #     print("!")
        #     print(f'Anomaly score for x={x:.3f}: {hst.score_one(features):.3f}')

    # print(len(anomaly_scores))
    # print(len(ts_obj.dataframe))

    # the outlier indices are based on filled_df where missing time steps are filled with nans
    # we need to make them be based off of ts_obj.dataframe where missing time steps are not present
    if ts_obj.miss:
        filled_df["anomaly scores"] = anomaly_scores
        # remove nans
        filled_df = filled_df.dropna()
        anomaly_scores = filled_df["anomaly scores"]

    # print(len(anomaly_scores))
    # print(len(ts_obj.dataframe))


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