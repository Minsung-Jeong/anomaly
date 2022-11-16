import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
import statsmodels.api as sm
import warnings
from itertools import product
from datetime import datetime

df = pd.read_csv("C://data_minsung/kaggle/bitcoin/bitcoin.csv")


df.Timestamp = pd.to_datetime(df.Timestamp, unit='s')

df.set_index('Timestamp', inplace=True)


df = df.resample('D').mean()
df_month =df.resample('M').mean()
df_quart = df.resample('Q-DEC').mean()
df_year = df.resample('Y').mean()


df.columns
# plot data

fig = plt.figure(figsize=[16,8])

plt.subplot(221)
plt.title('Daily price, USD', fontsize=20)
plt.plot(df["Weighted_Price"], label='Daily')
plt.legend()

plt.subplot(222)
plt.title('Monthly price, USD', fontsize=20)
plt.plot(df_month["Weighted_Price"], label='Monthly')
plt.legend()


plt.subplot(223)
plt.title('Quarter price, USD', fontsize=20)
plt.plot(df_quart["Weighted_Price"], label='Quarters')
plt.legend()

plt.subplot(224)
plt.title('Yearly price, USD', fontsize=20)
plt.plot(df_year["Weighted_Price"], label='Yearly')


plt.figure(figsize=[15,7])
sm.tsa.seasonal_decompose(df_month['Weighted_Price']).plot()
print("Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(df_month.Weighted_Price)[1])
plt.show()
# Box-cox Transformations
df_month['Weighted_Price_box'], lmbda = stats.boxcox(df_month.Weighted_Price)
# check stationary with Augmented Dickey-Fuller Test
# : if p-value is not less than 0.05(fail to reject h0) - time series is non-stationary
print("Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(df_month.Weighted_Price)[1])

# Seasonal differentiation
df_month['prices_box_diff'] = df_month['Weighted_Price_box'] - df_month.Weighted_Price_box.shift(12)
print("Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(df_month.prices_box_diff[12:])[1])

# Regular differentiation
df_month['prices_box_diff2'] = df_month['prices_box_diff']- df_month['prices_box_diff'].shift(1)
plt.figure(figsize=(15,7))

# stl decomposition
sm.tsa.seasonal_decompose(df_month['prices_box_diff2'][13:]).plot()


# Initial approximation of parameters using Autocorrelation and Partial Autocorrelation Plots
plt.figure(figsize=(15,7))
ax = plt.subplot(211)
# acf = auto-correlation function
sm.graphics.tsa.plot_acf(df_month.prices_box_diff2[13:].values.squeeze(), lags=48, ax=ax)
ax = plt.subplot(212)
# pacf = partial auto-correlation function
sm.graphics.tsa.plot_pacf(df_month.prices_box_diff2[13:].values.squeeze(), lags=48, ax=ax)
plt.tight_layout()
plt.show()


# Initial approximation of parameters
Qs = range(0, 2)
qs = range(0, 3)
Ps = range(0, 3)
ps = range(0, 3)
D=1
d=1
parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
len(parameters_list)

# Model Selection
results = []
best_aic = float("inf")
warnings.filterwarnings('ignore')
for param in parameters_list:
    try:
        model=sm.tsa.statespace.SARIMAX(df_month.Weighted_Price_box, order=(param[0], d, param[1]),
                                        seasonal_order=(param[2], D, param[3], 12)).fit(disp=-1)
    except ValueError:
        print('wrong parameters:', param)
        continue
    aic = model.aic
    if aic < best_aic:
        best_model = model
        best_aic = aic
        best_param = param
    results.append([param, model.aic])

# Best Models
result_table = pd.DataFrame(results)
result_table.columns = ['parameters', 'aic']
print(result_table.sort_values(by = 'aic', ascending=True).head())
print(best_model.summary())