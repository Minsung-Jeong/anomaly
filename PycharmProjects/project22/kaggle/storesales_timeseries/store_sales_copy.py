import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from sklearn.linear_model import LinearRegression
from pandas import date_range
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

# Model 1 (trend)
from sklearn.linear_model import ElasticNet, Lasso, Ridge

# Model 2
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import RegressorChain
import warnings
import os
# switch off the warnings
warnings.filterwarnings("ignore")

os.getcwd()
os.chdir("C://data_minsung/kaggle/store_sales")
# 2. Read data
df_holidays = pd.read_csv('./holidays_events.csv', header = 0)
df_oil = pd.read_csv('./oil.csv', header = 0)
df_stores = pd.read_csv('./stores.csv', header = 0)
df_trans = pd.read_csv('./transactions.csv', header = 0)

df_train = pd.read_csv('./train.csv', header = 0)
df_test = pd.read_csv('./test.csv', header = 0)

df_holidays['date'] = pd.to_datetime(df_holidays['date'], format = "%Y-%m-%d")
df_oil['date'] = pd.to_datetime(df_oil['date'], format = "%Y-%m-%d")
df_trans['date'] = pd.to_datetime(df_trans['date'], format = "%Y-%m-%d")
df_train['date'] = pd.to_datetime(df_train['date'], format = "%Y-%m-%d")
df_test['date'] = pd.to_datetime(df_test['date'], format = "%Y-%m-%d")


fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(25,15))
df_oil.plot.line(x="date", y="dcoilwtico", color='b', title ="dcoilwtico", ax = axes, rot=0)
plt.show()

def grouped(df, key, freq, col):
    """ GROUP DATA WITH CERTAIN FREQUENCY """
    df_grouped = df.groupby([pd.Grouper(key=key, freq=freq)]).agg(mean = (col, 'mean'))
    df_grouped = df_grouped.reset_index()
    return df_grouped

def add_time(df, key, freq, col):
    """ ADD COLUMN 'TIME' TO DF """
    df_grouped = grouped(df, key, freq, col)
    df_grouped['time'] = np.arange(len(df_grouped.index))
    column_time = df_grouped.pop('time')
    df_grouped.insert(1, 'time', column_time)
    return df_grouped


# check grouped data
df_grouped_trans_w = grouped(df_trans, 'date', 'W', 'transactions')
df_grouped_train_w = add_time(df_train, 'date', 'W', 'sales')
df_grouped_train_m = add_time(df_train, 'date', 'M', 'sales')

df_grouped_train_w.head() # check results


# regression 그리기
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(30,20))

# TRANSACTIONS (WEEKLY)
axes[0].plot('date', 'mean', data=df_grouped_trans_w, color='grey', marker='o')
axes[0].set_title("Transactions (grouped by week)", fontsize=20)

# SALES (WEEKLY)
axes[1].plot('time', 'mean', data=df_grouped_train_w, color='0.75')
axes[1].set_title("Sales (grouped by week)", fontsize=20)
# linear regression
axes[1] = sns.regplot(x='time',
                      y='mean',
                      data=df_grouped_train_w,
                      scatter_kws=dict(color='0.75'),
                      ax = axes[1])

# SALES (MONTHLY)
axes[2].plot('time', 'mean', data=df_grouped_train_m, color='0.75')
axes[2].set_title("Sales (grouped by month)", fontsize=20)
# linear regression
axes[2] = sns.regplot(x='time',
                      y='mean',
                      data=df_grouped_train_m,
                      scatter_kws=dict(color='0.75'),
                      line_kws={"color": "red"},
                      ax = axes[2])

plt.show()



# def add_lag(df, key, freq, col, lag):
#     """ ADD LAG """
#     df_grouped = grouped(df, key, freq, col)
#     name = 'Lag_' + str(lag)
#     df_grouped['Lag'] = df_grouped['mean'].shift(lag)
#     return df_grouped
#
# df_grouped_train_w_lag1 = add_lag(df_train, 'date', 'W', 'sales', 1)
# df_grouped_train_m_lag1 = add_lag(df_train, 'date', 'W', 'sales', 1)
#
# df_grouped_train_w_lag1.head() # check data
#
# # 칼럽 데이터들을 bar graph로 보여줌
# def plot_stats(df, column, ax, color, angle):
#     """ PLOT STATS OF DIFFERENT COLUMNS """
#     count_classes = df[column].value_counts()
#     ax = sns.barplot(x=count_classes.index, y=count_classes, ax=ax, palette=color)
#     ax.set_title(column.upper(), fontsize=18)
#     for tick in ax.get_xticklabels():
#         tick.set_rotation(angle)
#
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
# fig.autofmt_xdate() # x축 feature 위치 정돈해주는 함수
# fig.suptitle("Stats of df_holidays".upper())
# plot_stats(df_holidays, "type", axes[0], "pastel", 45)
# plot_stats(df_holidays, "locale", axes[1], "rocket", 45)
# plt.show()
#
# # trend & moving average 시각화
# def plot_moving_average(df, key, freq, col, window, min_periods, ax, title):
#     df_grouped = grouped(df, key, freq, col)
#     moving_average = df_grouped['mean'].rolling(window=window, center=True, min_periods=min_periods).mean()
#     ax = df_grouped['mean'].plot(color='0.75', linestyle='dashdot', ax=ax)
#     ax = moving_average.plot(linewidth=3, color='g', ax=ax)
#     ax.set_title(title, fontsize=18)
#
# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(30,20))
# plot_moving_average(df_trans, 'date', 'W', 'transactions', 7, 4, axes[0], 'Transactions Moving Average')
# plot_moving_average(df_train, 'date', 'W', 'sales', 7, 4, axes[1], 'Sales Moving Average')
# plt.show()




"""
5. Hybrid Models
"""
store_sales = df_train.copy()

store_sales['date'] = store_sales.date.dt.to_period('D')
store_sales.head()
store_sales = store_sales.set_index(['store_nbr', 'family', 'date']).sort_index()
store_sales.head()
family_sales = (
    store_sales
    .groupby(['family', 'date'])
    .mean()
    .unstack('family')
    .loc['2017']
)

# we'll add fit and predict methods to this minimal class
class BoostedHybrid:
    def __init__(self, model_1, model_2):
        self.model_1 = model_1
        self.model_2 = model_2
        self.y_columns = None  # store column names from fit method

    def fit(self, X_1, X_2, y):
        # train model_1
        self.model_1.fit(X_1, y)
        # make predictions
        y_fit = pd.DataFrame(self.model_1.predict(X_1), index=X_1.index, columns=y.columns,)

        # compute residuals
        y_resid = y - y_fit
        y_resid = y_resid.stack().squeeze() # wide to long

        # train model_2 on residuals
        self.model_2.fit(X_2, y_resid)

        # save column names for predict method
        self.y_columns = y.columns
        # Save data for question checking
        self.y_fit = y_fit
        self.y_resid = y_resid

    def predict(self, X_1, X_2):
        # Predict with model_1
        y_pred = pd.DataFrame(
            self.model_1.predict(X_1),
            index=X_1.index, columns=self.y_columns,
        )
        y_pred = y_pred.stack().squeeze()  # wide to long

        # Add model_2 predictions to model_1 predictions
        y_pred += self.model_2.predict(X_2)

        return y_pred.unstack()

# 데이터 준비
# Target series
y = family_sales.loc[:, 'sales']


# X_1: Features for Linear Regression
dp = DeterministicProcess(index=y.index, order=1)
X_1 = dp.in_sample()


# X_2: Features for XGBoost
X_2 = family_sales.drop('sales', axis=1).stack()  # onpromotion feature

# Label encoding for 'family'
le = LabelEncoder()  # from sklearn.preprocessing
X_2 = X_2.reset_index('family')
X_2['family'] = le.fit_transform(X_2['family'])

# Label encoding for seasonality
X_2["day"] = X_2.index.day  # values are day of the month