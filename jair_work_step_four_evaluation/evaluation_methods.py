from sklearn.metrics import f1_score
import numpy as np
import pandas as pd

def get_nab_score(dataset, method):
	nab_score_df = pd.read_csv("nab_scores.csv")
	narrow_df = nab_score_df[(nab_score_df["anomaly detection method"] == method) & (nab_score_df["dataset"] == dataset)]
	return narrow_df["standard nab score"].values[0]


def point_f_score(threshold, scores, true):
	predict = []
	for score in scores:
		if score >= threshold:
			predict.append(1)
		else:
			predict.append(0)
	return f1_score(true, predict, average='binary')


def windowed_f_score(threshold, scores, true):
	predict = []
	for score in scores:
		if score >= threshold:
			predict.append(1)
		else:
			predict.append(0)
	num_gt_outliers = int(sum(true))
	predict_window = get_window(predict, num_gt_outliers=num_gt_outliers, window_size="default")
	true_window = get_window(true, num_gt_outliers=num_gt_outliers, window_size="default")
	return f1_score(true_window, predict_window, average='binary')




def get_window(a_list, num_gt_outliers=2, window_size="default"):
    """
    Input:

    a_list
    e.g. [0,0,1,0,1...]

    window size for anomaly detection
    a WINDOW is now anomalous if a point in it is anomalous

    num_gt_outliers = number of ground truth outliers
    necessary to determine default window size

    Output:

    a_list windowed
    e.g. if window_size = 2 and a_list = [0,0,0,1,1,0,0,...]
    we would return [0,1,1]
    """
    if num_gt_outliers == 0:
            raise ValueError("Cannot calculate window precision and recall if no outliers.")
    # determine window size
    if window_size == "default":
        window_size = int((.1 * len(a_list)) / num_gt_outliers)
    else:
        if window_size > len(a_list):
            raise ValueError("Given window size is too large.")
    # print("Window size", window_size)
    number_of_windows = int(len(a_list) / window_size)
    chunked_list = np.array_split(a_list, number_of_windows)
    windowed_list = []
    for chunk in chunked_list:
        if 1 in chunk:
            windowed_list.append(1)
        else:
            windowed_list.append(0)
    return windowed_list
