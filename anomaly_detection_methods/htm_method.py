import matplotlib.pyplot as plt
import sys  
sys.path.append("../characteristics")  
import characteristics_helpers as ch
import pandas as pd


def convert_into_htm_format(ts_obj):
    if ts_obj.get_length() < 400:
    	raise ValueError("HTM requires at least 400 time steps.")
    if ts_obj.miss:
        ref_date_range = ch.get_ref_date_range(ts_obj.dataframe, ts_obj.dateformat, ts_obj.timestep)
        gaps = ref_date_range[~ref_date_range.isin(ts_obj.dataframe["timestamp"])]
        filled_df = ch.fill_df(ts_obj.dataframe, ts_obj.timestep, ref_date_range, "fill_nan")
        # print("NaNs exist?: ",filled_df['value'].isnull().values.any())
        df_to_write = pd.DataFrame({"timestamp": filled_df.index, "value": filled_df["value"].values})
    else:
        df_to_write = pd.DataFrame({"timestamp": ts_obj.dataframe["timestamp"], "value": ts_obj.dataframe["value"]})

    df_to_write.to_csv("htm_input_csvs/" + ts_obj.name ,index=False)

def htm_method(ts_obj, outer_path, plot_anomaly_score=False):
	no_csv_name = ts_obj.name[:-4]
	htm_results = pd.read_csv(outer_path + "/htm_results_" + no_csv_name + "_value.csv", header=0)
	anomaly_scores = htm_results["raw_anomaly_score"].values
	times = pd.read_csv(outer_path + "/htm_times.csv", header=0)
	time = times[times["Name"] == ts_obj.name]["HTM Time"].values[0]
	
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
	        "Time":time}

