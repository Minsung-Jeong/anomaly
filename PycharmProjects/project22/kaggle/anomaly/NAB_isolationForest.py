import numpy as np
import pandas as pd

import matplotlib.dates as mdates
from sklearn.ensemble import IsolationForest

# https://www.kaggle.com/code/joshuaswords/time-series-anomaly-detection

df = pd.read_csv("C://data_minsung/kaggle/nab/realKnownCause/realKnownCause/nyc_taxi.csv", parse_dates=['timestamp'])
def check_df(df):
    print(df.isnull().sum(), '\n')
    print(df.dtypes, '\n')
    print(df.describe())

check_df(df)