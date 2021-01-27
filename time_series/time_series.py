import pandas as pd
import numpy as np
import sys  
sys.path.append("../characteristics")  
import characteristics
import characteristics_helpers

class TimeSeries:

    def __init__(self, dataframe, timestep, dateformat, name=None):

        if ("value" not in dataframe) or ("timestamp" not in dataframe):
            raise ValueError("The given dataframe must have 'value' and 'timestamp' columns.")

        dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], format=dateformat)
        dataframe = characteristics_helpers.remove_duplicate_time_steps(start_date=dataframe["timestamp"].values[0], end_date=dataframe["timestamp"].values[-1], time_step_size=timestep, given_dataframe=dataframe, value="value")

        # the following are automatically initialized by the call
        self.dataframe = dataframe
        self.timestep = timestep
        self.dateformat = dateformat
        self.name = name

        # the following must first be set
        self.miss = None
        self.seasonality = None
        self.period = None
        self.concept_drift = None
        self.trend = None
        self.trend_type = None

    def get_length(self):
        """Get length of dataframe"""
        return len(self.dataframe)

    def get_probationary_index(self):
        return int(.15 * len(self.dataframe))

    def get_mean(self):
        """Get mean of dataframe"""
        return np.mean(self.dataframe['value'])

    def get_min(self):
        """Get min of dataframe"""
        return np.min(self.dataframe['value'])

    def get_max(self):
        """Get max of dataframe"""
        return np.max(self.dataframe['value'])

    def get_median(self):
        """Get median of dataframe"""
        return np.median(self.dataframe['value'])

    def get_variance(self):
        """Get variance of dataframe"""
        return np.var(self.dataframe['value'])

    def get_std(self):
        """Get standard deviation of dataframe"""
        return np.std(self.dataframe['value'])

    def get_timestep(self):
        """Get time step size of dataframe"""
        return self.timestep

    def get_dateformat(self):
        """Get the string date format of dataframe"""
        return self.dateformat

    def set_trend(self):
        self.trend, self.trend_type = characteristics.has_trend(self.dataframe)

    def get_trend(self):
        return self.trend

    def get_trend_type(self):
        return self.trend_type

    def set_seasonality(self):
        s, p = characteristics.has_seasonality(self.dataframe)
        self.seasonality = s
        self.period = p

    def get_period(self):
        return self.period

    def get_seasonality(self):
        return self.seasonality

    def set_concept_drift(self):
        self.concept_drift = characteristics.has_concept_drift(self.dataframe)

    def get_concept_drift(self):
        return self.concept_drift

    def set_miss(self, fill):
        self.miss, self.dataframe = characteristics.has_miss(self.dataframe, self.dateformat, self.timestep, fill)

    def get_miss(self):
        return self.miss

    def get_how_many_miss(self):
        return characteristics.how_many_miss(self.dataframe, self.dateformat, self.timestep)
