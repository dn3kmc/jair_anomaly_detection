import pandas as pd

def create_df(values):
    # given a list create a df with some arbitrarily chosen dates
    rng = pd.date_range('2000-01-01', periods=len(values), freq='T')
    df = pd.DataFrame({ 'timestamp': rng, 'value': values})
    return df


def rename_df(df, timestamp_string, value_string):
    # given a df, rename the columns to "timestamp" and "value"
    new_df = pd.DataFrame({"timestamp": df[timestamp_string], "value": df[value_string]})
    return new_df
