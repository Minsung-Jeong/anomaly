import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
os.chdir('C://data_minsung/finance/Qraft')
etfs = pd.read_csv('./indexPortfolio/etfs.csv').set_index('Date')
etfs_ret = etfs.pct_change(1).iloc[1:]

# 무위험 시장에 따른 Tangency portfolios 생성
def get_tan_weight(returns):

    mu = np.matrix(np.mean(returns, axis=1)).T
    cov = np.matrix(np.cov(returns))
    cov_inv = np.linalg.inv(cov)

    numerator = np.dot(cov_inv, mu)
    one_matrix = np.matrix(np.ones(mu.shape[0]))
    denominator = np.dot(one_matrix, cov_inv).dot(mu)

    tan_weight = numerator / denominator[0,0]
    weight_df = pd.DataFrame(tan_weight, index=asset_n)
    return weight_df

def re_weight(weight_df):
    asset_n = weight_df.index.values
    # 최솟값이 0보다 작으면 그 값의 절대값만큼 전체에 더해줌
    if weight_df.values.min() < 0:
        weight_df = weight_df - weight_df.min()
    weight_df = weight_df/ weight_df.sum()
    weight_df = weight_df.sort_values(by=0, ascending=False)
    weight_val = weight_df.values
    # 0.25보다 큰 비중에 대한 조정(0.25보다 큰 값만큼 다음 값에 넘겨줌)
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


# 월별 인덱스
etfs_ret.index = pd.to_datetime(etfs_ret.index)
add_ = etfs_ret.index[0]

last_idx = etfs_ret.resample('M').last().index
first_idx = last_idx + timedelta(days=1)
first_idx = list(first_idx)[:-1]
first_idx.insert(0, add_)

ret_li = []
for i in range(len(first_idx)-1):
    inputs = etfs_ret.apply(lambda x: x[first_idx[i]:last_idx[i+1]])
    inputs = inputs.dropna(axis=1).T
    asset_n = inputs.index.values

    weight_df = get_tan_weight(inputs)
    invest_ass = weight_df.index.values
    re_weighted = re_weight(weight_df)

    in_mu = etfs_ret.apply(lambda x: x[first_idx[i+1]:last_idx[i+1]])
    # in_mu = in_mu.dropna(axis=1)
    in_mu = in_mu[invest_ass]
    mu = np.matrix(np.mean(in_mu))

    total_return = mu @ re_weighted
    ret_li.append(total_return.values[0,0])

ret_df = pd.DataFrame(ret_li, index=last_idx[1:])
cum_ret_df = ret_df.add(1).cumprod().sub(1)

plt.plot(cum_ret_df)



def get_mdd(x):
    prc = pd.DataFrame((x+1).cumprod())
    DD = -(prc.cummax()-prc)
    MDD = DD.min()[0]
    return MDD, DD

# 압도적으로 낮은 MDD의 원인은 비중을 0~25%로 제한했기 때문으로 판단함
MDD, DD = get_mdd(ret_df)

