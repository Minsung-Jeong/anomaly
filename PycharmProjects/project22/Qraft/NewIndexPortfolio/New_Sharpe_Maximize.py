import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

os.chdir('C://data_minsung/finance/Qraft')

# etfs.index[0].strftime('%d')
def get_tan_weight(returns, index):

    mu = np.matrix(np.mean(returns, axis=1)).T
    cov = np.matrix(np.cov(returns))
    cov_inv = np.linalg.inv(cov)

    numerator = np.dot(cov_inv, mu)
    one_matrix = np.matrix(np.ones(mu.shape[0]))
    denominator = np.dot(one_matrix, cov_inv).dot(mu)

    tan_weight = numerator / denominator[0,0]
    weight_df = pd.DataFrame(tan_weight, index=index)
    return weight_df


"""
weight 구성 로직 : 많은 자산에 골고루 배분할 수 있는 방식 
1. 내림차순으로 sort
2. 가장 작음 음수의 절댓값만큼 모든 value에 값 더함
3. 총합이 1을 넘으므로 sum값으로 나눔
4. x > 0.25 면 diff = x - 0.25
5. diff / (len(val) - (i+1)) 를 뒤의 모든 값에 더해줌
"""
def re_weight(weight):
    temp = weight.sort_values(by=0, ascending=False)
    temp = temp-temp.iloc[-1]
    temp = temp / temp.sum()
    temp_val = temp.values
    temp_idx = temp.index

    new_temp = []
    for i in range(len(temp_val)):
        if temp_val[i][0] > 0.25:
            diff = temp_val[i][0] - 0.25
            denom = len(temp_val) - (i+1)
            new_temp.append(0.25)
            if denom != 0 and i+1 < len(temp_val):
                temp_val[i+1:] = temp_val[i+1:] + diff/denom
        else:
            new_temp.append(temp_val[i][0])
    return pd.Series(new_temp, index=temp_idx)

etfs = pd.read_csv('./indexPortfolio/etfs.csv').set_index('Date')
etfs.index = pd.to_datetime(etfs.index, format='%Y-%m-%d')
etfs_ret = etfs.pct_change().iloc[1:]

last_d = etfs.resample('M').last().index
first_d = last_d[:-1] + timedelta(days=1)
first_d = first_d.insert(0, etfs.index[0])

# pct_change은 월단위 차이를 반영하도록 구성
monthly_return = etfs.resample('M').last().pct_change().iloc[1:]

# Sharpe max input
# 12개월에 대한 관측치를 이용하여 sharpe max 도출(기간이 너무 길면 적시성 bad)
# last_d 와 monthly_return의 길이가 다르기 때문에 코드상 혼란야기
result = []
for i in range(12, len(last_d)-1):

    start = last_d[i-11]
    end = last_d[i+1]

    sharpe_inputs = monthly_return[start:end].dropna(axis=1).T
    weight = get_tan_weight(sharpe_inputs, sharpe_inputs.index)
    # 음수 없고, 25이하로 하도록 weight 조정
    re_weighted = re_weight(weight)

    # profit of rebalance
    profit = monthly_return.iloc[i].dropna()

    temp_profit = 1
    for asset in re_weighted.index:
        # profit은 되는데 re_weighted는 안됨 - 이유?
        temp_profit += profit[asset] * re_weighted[asset]
    result.append(temp_profit)


port_profit = pd.Series(result, index=last_d[13:])


temp_cum = [100]
for x in result:
    temp_cum.append(temp_cum[-1]*x)


def get_mdd(x):
    prc = pd.DataFrame(x.cumprod())
    DD = -(prc.cummax()-prc)
    MDD = DD.min()[0]
    return MDD, DD

port_profit.cumprod()

# sharpe maximize + 25%이하 비중 조정이라는 조건제한을 통해 안정성이 굉장히 높아지는 효과
mdd, dd = get_mdd(port_profit)