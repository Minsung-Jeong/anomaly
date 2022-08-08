import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import solvers
import cvxpy as cvx

# MVO(Mean-Variance Optimization) - Tangency Portfolio
# etfs 결측치는 가장 가까운 값으로 대체(nn에서는 평균값 or 최빈값)
os.chdir('C://data_minsung/finance/Qraft')
etfs = pd.read_csv('./indexPortfolio/etfs.csv')

# Nan값은 가장 가까운 날의 값으로 대체 - 모두 1000인 것을 확인
etfs = etfs.fillna(float(1000)).set_index('Date')
etfs_ret = etfs.pct_change(1).dropna()
asset_n = etfs.columns.values

returns = etfs_ret.iloc[:30].values.T


mean_ret = np.mean(returns, axis=1)
cov_ret = np.cov(returns)
n = mean_ret.shape[0]
one_array = np.ones(n)

# 무위험 시장에 따른 Tangency portfolios 생성
rf = 0.0001
mu = np.matrix(mean_ret).T
cov = np.matrix(cov_ret)
cov_inv = np.linalg.inv(cov)

numerator = np.dot(cov_inv, mu)
one_matrix = np.matrix(np.ones(mu.shape[0]))
denominator = np.dot(one_matrix, cov_inv).dot(mu)

tan_weight = numerator / denominator[0,0]
weight_df = pd.DataFrame(tan_weight, index=asset_n)

def re_weight(weight_df):
    if min(weight_df)[0,0] < 0:
        weight_df = weight_df - weight_df.min()

    weight_df = weight_df/ weight_df.sum()
    weight_df = weight_df.sort_values(by=0, ascending=False)





# 아래 로직 끼워넣기 위해서는 1. pandas 데이터에 대해 적용, 2. sort한 것 원래대로 하는 것것
tep = [0.3, 0.25, 0.21, 0.2, 0.04]

for i in range(len(temp)):

    if temp[i] > 0.25 and i != len(temp)-1:
        diff = temp[i] - 0.25
        temp[i+1] = temp[i+1] + diff
        temp[i] = 0.25

    if temp[i] > 0.25 and i == len(temp)-1:
        temp[i] = 0.25

