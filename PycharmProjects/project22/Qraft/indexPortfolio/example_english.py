import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import solvers
import cvxpy as cvx

from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.expected_returns import ema_historical_return
from pypfopt.risk_models import exp_cov
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.plotting import plot_efficient_frontier
from pypfopt.plotting import plot_weights
from pypfopt.cla import CLA


# MVO(Mean-Variance Optimization) - Tangency Portfolio
# etfs 결측치는 가장 가까운 값으로 대체(nn에서는 평균값 or 최빈값)
os.chdir('C://data_minsung/finance/Qraft')
etfs = pd.read_csv('./indexPortfolio/etfs.csv')

# Nan값은 가장 가까운 날의 값으로 대체 - 모두 1000인 것을 확인
# Nan_ticker = ['TLT','EMB','Cash','VWO','DBC']
# fill_val = []
# for ticker in Nan_ticker:
#     for x in etfs[ticker]:
#         if not math.isnan(x):
#             fill_val.append(x)
#             break

etfs = etfs.fillna(float(1000))
etfs = etfs.set_index('Date')
etfs_ret = etfs.pct_change(1).dropna().T
etfs_cumret = etfs_ret.add(1).cumprod().sub(1)*100

train = etfs_ret

np.mean(train)


mu = expected_returns.ema_historical_return(train, returns_data = True, span = 500)
Sigma = risk_models.exp_cov(train, returns_data = True, span = 180)

np.shape(Sigma)


ret_ef = np.arange(0, 0.879823, 0.01)
vol_ef = []
for i in ret_ef:
    ef = EfficientFrontier(mu, sigma)