dataset_name = "Twitter_volume_GOOG"
date_format = "%Y-%m-%d %H:%M:%S"
# %Y-%m-%d %H:%M:%S
# %Y-%m-%d
# %Y-%m
time_step = "5min"
# 5min
# 30min
# 1H
# 1D
# 1MS
path = dataset_name + ".csv"






import pandas as pd
import math
import numpy as np

import sys  
sys.path.append("../characteristics")  
import characteristics


data = pd.read_csv(path,header=0)

data["timestamp"] = pd.to_datetime(data["timestamp"], 
                                   format=date_format)


bool_value, df = characteristics.has_miss(data, date_format, time_step, fill=False)

if bool_value:
	print("!!!!")
	print("has missing")
	bool_value, df = characteristics.has_miss(data, date_format, time_step, fill=True)
	df.to_csv(dataset_name + "_filled.csv")



# warning: this does not fill out outlier column for you
# also timestamp is not named