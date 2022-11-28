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
df_year = df.resample('Y').mean()

# plot data
f, axes = plt.subplots(3,1)
f.set_size_inches((10,6))
axes[0].plot(df["Weighted_Price"], label='Daily')
axes[0].legend()
axes[1].plot(df_month["Weighted_Price"], label='Monthly')
axes[1].legend()
axes[2].plot(df_year["Weighted_Price"], label='Yearly')
axes[2].legend()
plt.show()

# seasonal decompose of month price
# 육안으로 봤을 때 station 하지 않음
sm.tsa.seasonal_decompose(df_month['Weighted_Price']).plot()
# check stationary with Augmented Dickey-Fuller Test
# : if p-value is not less than 0.05(fail to reject h0) - time series is non-stationary
print("Augmented Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(df_month.Weighted_Price)[1])


# Box-cox Transformations
# Box-cox T doesn't work
df_month['Weighted_Price_box'], lmbda = stats.boxcox(df_month.Weighted_Price)
print("Augmented Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(df_month['Weighted_Price_box'])[1])

# Seasonal differentiation
# Seasonal differentiation doesn't work
df_month['prices_box_diff'] = df_month['Weighted_Price_box'] - df_month.Weighted_Price_box.shift(12)
print("Augmented Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(df_month.prices_box_diff[12:])[1])

# Regular differentiation(Double Differentiation)
# It works!
df_month['prices_box_diff2'] = df_month['prices_box_diff']- df_month['prices_box_diff'].shift(1)
print("Augmented Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(df_month['prices_box_diff2'][13:])[1])
plt.figure(figsize=(15,7))

# stl decomposition of double differentiation
sm.tsa.seasonal_decompose(df_month['prices_box_diff2'][13:]).plot()


# Initial approximation of parameters using Autocorrelation and Partial Autocorrelation Plots
plt.figure(figsize=(15,7))
ax = plt.subplot(211)
# acf = auto-correlation function
sm.graphics.tsa.plot_acf(df_month['prices_box_diff2'][13:].values.squeeze(), lags=48, ax=ax)
ax = plt.subplot(212)
# pacf = partial auto-correlation function
sm.graphics.tsa.plot_pacf(df_month['prices_box_diff2'][13:].values.squeeze(), lags=48, ax=ax)
plt.tight_layout()
plt.show()


# Initial approximation of parameters
p = P = q = range(0, 3)
Q = range(0, 2)
D = d = 1

parameters = list(product(p, q, P, Q))
# pdq = list(product(p,d,q))
# pdq_seasonal = [(x[0],x[1],x[2],12) for x in list(pdq).copy()]

# Model Selection
results = []
best_aic = float("inf")
warnings.filterwarnings('ignore')

for param in parameters:
    try:
        temp = sm.tsa.statespace.SARIMAX(df_month.Weighted_Price_box, order=(param[0], d, param[1]),
                                          seasonal_order=(param[2], D, param[3], 12))
        model = temp.fit(disp=-1)

        if model.aic < best_aic:
            best_model = model
            best_aic = model.aic
            best_pdq = param
        results.append([param, model.aic])
    except ValueError:
        continue


# Best Models
result_table = pd.DataFrame(results)
result_table.columns = ['parameters', 'aic']
print(result_table.sort_values(by = 'aic', ascending=True).head())
print(best_model.summary())



# Analysis of residues
# STL-decomposition
plt.figure(figsize=(15,7))
plt.subplot(211)
best_model.resid[13:].plot()
plt.ylabel(u'Residuals')
ax = plt.subplot(212)
sm.graphics.tsa.plot_acf(best_model.resid[13:].values.squeeze(), lags=48, ax=ax)

print("Augmented Dickey–Fuller test:: p=%f" % sm.tsa.stattools.adfuller(best_model.resid[13:])[1])

plt.tight_layout()
plt.show()


# Prediction
# Inverse Box-Cox Transformation Function
def invboxcox(y,lmbda):
   if lmbda == 0:
      return(np.exp(y))
   else:
      return(np.exp(np.log(lmbda*y+1)/lmbda))

# Prediction
df_month2 = df_month[['Weighted_Price']]
date_list = [datetime(2017, 6, 30), datetime(2017, 7, 31), datetime(2017, 8, 31), datetime(2017, 9, 30),
             datetime(2017, 10, 31), datetime(2017, 11, 30), datetime(2017, 12, 31), datetime(2018, 1, 31),
             datetime(2018, 1, 28)]
future = pd.DataFrame(index=date_list, columns= df_month.columns)
df_month2 = pd.concat([df_month2, future])
df_month2['forecast'] = invboxcox(best_model.predict(start=0, end=75), lmbda)
plt.figure(figsize=(15,7))
df_month2.Weighted_Price.plot()
df_month2.forecast.plot(color='r', ls='--', label='Predicted Weighted_Price')
plt.legend()
plt.title('Bitcoin exchanges, by months')
plt.ylabel('mean USD')
plt.show()
