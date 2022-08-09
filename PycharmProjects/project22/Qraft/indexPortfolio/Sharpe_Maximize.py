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
etfs = etfs.set_index('Date')
etfs = etfs.fillna(float(1000))

etfs_ret = etfs.pct_change(1).dropna()
asset_n = etfs_ret.columns.values

returns = etfs_ret.iloc[:40].values.T


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
    if weight_df.values.min() < 0:
        weight_df = weight_df - weight_df.min()
    weight_df = weight_df/ weight_df.sum()
    weight_df = weight_df.sort_values(by=0, ascending=False)
    weight_val = weight_df.values
    for i in range(len(weight_val)):
        if weight_val[i] > 0.25 and i != len(weight_val)-1:
            diff = weight_df.iloc[i] - 0.25
            weight_df.iloc[i + 1] = weight_df.iloc[i + 1] + diff
            weight_df.iloc[i] = 0.25
        if weight_val[i] > 0.25 and i == len(weight_val) - 1:
            weight_df.iloc[i] = 0.25
    temp = []
    for A in asset_n:
        temp.append(weight_df.T[A].values[0])
    weight_df = pd.DataFrame(temp, index=asset_n)
    return weight_df

re_weighted = re_weight(weight_df)


# 날짜별로 가져오기(결측치 어떻게 처리?)