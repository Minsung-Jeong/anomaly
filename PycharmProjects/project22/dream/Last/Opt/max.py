import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 목표 : 이상적인 sharpe Maximized 포트폴리오 구축 및 결과분석
def get_tan_weight(returns):
    cols = returns.columns
    mu = np.matrix(np.mean(returns)).T
    cov = np.matrix(np.cov(returns.T))
    cov_inv = np.linalg.inv(cov)

    numerator = np.dot(cov_inv, mu)
    one_matrix = np.matrix(np.ones(mu.shape[0]))
    denominator = np.dot(one_matrix, cov_inv).dot(mu)
    tan_weight = numerator / denominator[0, 0]
    weight_df = pd.DataFrame(tan_weight, index=cols)
    return weight_df

"""
weight 구성 로직 : 많은 자산에 골고루 배분할 수 있는 방식 
1. 내림차순으로 sort
2. 가장 작은 값의 절댓값만큼 모든 값에 더함
3. 총합이 1을 넘으므로 sum으로 나눔
4. 0.25 넘는 값은 0.25로 변경 뒤 차잇값(diff)만큼 뒤의 값에 나눠서 더해줌
5. 4번에 대한 값 = diff / (len(val) - (i+1)) 
"""
def re_weight(weight):
    # 내림차순 정렬 -> 최솟값의 절댓값 모든 값에 더하기(음수 제거) ->
    temp = weight.sort_values(by=0, ascending=False)
    temp = temp + abs(min(temp.values)[0])
    temp = temp / temp.sum()
    temp_val = temp.values
    temp_idx = temp.index

    new_temp = []
    for i in range(len(temp_val)):
        if temp_val[i][0] > 0.25:
            diff = temp_val[i][0] - 0.25
            denom = len(temp_val) - (i + 1)
            new_temp.append(0.25)
            if denom != 0 and i + 1 < len(temp_val):
                temp_val[i + 1:] = temp_val[i + 1:] + diff / denom
        else:
            new_temp.append(temp_val[i][0])
    return pd.Series(new_temp, index=temp_idx)

def get_data(url):

    etfs = pd.read_csv(url).set_index('Date')
    etfs.index = pd.to_datetime(etfs.index)

    # EMB 결측치가 3413으로 가장 많다
    etfs.isna().sum()
    etfs = etfs.dropna()

    ret_month = etfs.resample('M').last().pct_change()[1:]
    end_d = ret_month.index
    return ret_month, end_d
"""
일별 데이터(80/1/1~21/5/13) -> 결측치 제거(93/1/29 ~ 21/5/13)
월별 93/2/28~21/5/31
end_d = 매월 마지막(93/2/28~21/5/31),
"""
def execution(ret_month, end_d):
    # ex) 93/2/28~94/2/28(13개월) 데이터 이용해서,
    # 94년 1월 31일 결정(1 month forward looking)-1월31일~2월28일 사이의 profit
    result = []
    weight_df = pd.DataFrame((np.zeros_like(ret_month)), columns=ret_month.columns)
    for i in range(12, len(end_d)-1): #데이터 5월 13일까지 있으므로 마지막 인덱스는 생략
        s_idx = end_d[i-12]
        e_idx = end_d[i]
        sharpe_inputs = ret_month[s_idx: e_idx]

        weight = get_tan_weight(sharpe_inputs)

        # 25넘는 부분 내리고 음수는 양수화하는 re-weight
        re_weighted = re_weight(weight)

        weight_df.iloc[i] = re_weighted
        # profit
        profit = ret_month.iloc[i]

        temp_profit = 1
        for asset in re_weighted.index:
            temp_profit += profit[asset] * re_weighted[asset]
        result.append(temp_profit)
    return result, weight_df

def get_mdd(x):
    prc = pd.DataFrame(x.cumprod())
    DD = -(prc.cummax()-prc)
    MDD = DD.min()[0]
    return MDD, DD

os.chdir('C://data_minsung/finance/Qraft')
ret_month, end_d = get_data('./indexPortfolio/etfs.csv')
result, weight_df = execution(ret_month, end_d)

weight_df.index = end_d
weight_df = weight_df.iloc[12:-1]

# 누적 수익률
temp_cum = [100]
for x in result:
    temp_cum.append(temp_cum[-1]*x)

result_df = pd.DataFrame(index=end_d[12:-1])
result_df['Profit'] = result
result_df['Profit_Cum'] = temp_cum[1:]

port_profit = pd.Series(result, index=end_d[12:-1])
plt.plot(port_profit.cumprod())

mdd, dd = get_mdd(port_profit)

# 수익률, MDD 확인 및 시각화한 뒤 분석(weight 구성2가지-'내가한 것 vs 단순 4위까지만 25%'비교 )
result_df['Draw_Down'] = dd

# result_df.to_csv("./result/shapeMax.csv")
# weight_df.to_csv("./result/sharpeMax_port.csv")