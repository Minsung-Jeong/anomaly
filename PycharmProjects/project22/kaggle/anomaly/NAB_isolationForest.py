import numpy as np
import pandas as pd

import matplotlib.dates as mdates
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
# https://www.kaggle.com/code/joshuaswords/time-series-anomaly-detection

df = pd.read_csv("C://data_minsung/kaggle/nab/realKnownCause/realKnownCause/nyc_taxi.csv", parse_dates=['timestamp'])
def check_df(df):
    print(df.isnull().sum(), '\n')
    print(df.dtypes, '\n')
    print(df.describe())


# df_hourly = df.set_index('timestamp').resample('H').mean()
# df_daily = df.set_index('timestamp').resample('D').mean()
# df_weekly = df.set_index('timestamp').resample('W').mean()
#
# # 데이터 계절성 존재
# plt.plot(df_hourly)
# plt.plot(df_daily)
# plt.plot(df_weekly)


df_hourly = df.set_index('timestamp').resample('H').mean().reset_index()
df_daily = df.set_index('timestamp').resample('D').mean().reset_index()
df_weekly = df.set_index('timestamp').resample('W').mean().reset_index()


for DataFrame in [df_hourly, df_daily]:
    DataFrame['Weekday'] = (pd.Categorical(DataFrame['timestamp'].dt.strftime('%A'),
                                           categories=['Monday', 'Tuesday', 'Wednesday',
                                                       'Thursday','Friday', 'Saturday', 'Sunday']))

DataFrame['Hour'] = DataFrame['timestamp'].dt.hour
DataFrame['Day'] = DataFrame['timestamp'].dt.weekday
DataFrame['Month'] = DataFrame['timestamp'].dt.month
DataFrame['Year'] = DataFrame['timestamp'].dt.year
DataFrame['Month_day'] = DataFrame['timestamp'].dt.day
DataFrame['Lag'] = DataFrame['value'].shift(1)
DataFrame['Rolling_Mean'] = DataFrame['value'].rolling(7, min_periods=1).mean()
DataFrame = DataFrame.dropna()

# plot 관련은 집에서
    #to-do : 요일별, 시간별, 월별 추이
DataFrame[DataFrame['Weekday'] == 'Monday']
DataFrame[DataFrame['Weekday'] == 'Tuesday']


# Hour 데이터 요일별, 시간별 demand plot하기
df_hourly['Hour'] = df_hourly['timestamp'].dt.strftime('%H')
by_weekday = df_hourly.groupby( ['Hour', 'Weekday']).mean()['value'].unstack()

# 요일별 histogram 그리기
hist_by_weekday = df_hourly.groupby(['Weekday']).mean()

# 시간대별 plot 해주기
by_hour = df_hourly.groupby(['Hour']).mean()


temp = (df_hourly.join(df_hourly.groupby(['Hour','Weekday'])['value'].mean(),
                   on = ['Hour', 'Weekday'], rsuffix='_Average'))

df_hourly.groupby(['Hour'])['value'].mean()
df_hourly.groupby(['Weekday'])['value'].mean()

# more feature engineering
df_hourly = (df_hourly
             .join(df_hourly.groupby(['Hour','Weekday'])['value'].mean(),
                   on = ['Hour', 'Weekday'], rsuffix='_Average'))


df_daily = (df_daily
            .join(df_daily.groupby(['Hour','Weekday'])['value'].mean(),
                  on = ['Hour', 'Weekday'], rsuffix='_Average'))

df_hourly.tail()


# model : isolation forest tunning 과정, ensemble 과정


