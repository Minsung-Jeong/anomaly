import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

"""
data check
"""
def data_pre(directory):
    price = pd.read_csv(directory)
    price = price.set_index('Unnamed: 0')
    price.index = pd.to_datetime(price.index, format='%Y-%m-%d')

    # null value : B=46, N=14, P=97
    price.isnull().sum()

    # momentum score 구하기(1개월전/12개월전-1)
    obs, cols = np.shape(price)
    momentum = pd.DataFrame(np.zeros((obs, cols)), index=price.index, columns=price.columns)
    for col in price:
        for i in range(12, obs):
            momentum[col][i] = price[col][i-1] / price[col][i-12] - 1
    # 12개월, 1개월 전 데이터 이용하여 지금 모멘텀 구하므로 12개월치 데이터는 삭제
    momentum = momentum.iloc[12:]

    # 모멘텀 스코어 top5 도출
    top5 = []
    for i in range(len(momentum)-1): # 마지막 모멘텀은 필요 x - 모멘텀 통한 투자 다음달에 수익률 도출
        asset = momentum.iloc[i].sort_values(ascending=False)[:5]
        top5.append(asset.index.values)

    # 이전 달 모멘텀으로 다음 달 수익률 도출하므로 profit은 12+1개월 삭제하여 인덱스 맞춰줌
    profit = price.pct_change()[13:]+1
    price = price.iloc[13:]

    return price, profit, top5, momentum

"""
portfolio1 : rebalance top5 every month
"""
def port1(price, profit, top5):
    port_weight = pd.DataFrame(np.zeros((len(profit), len(profit.columns))), index=profit.index, columns=profit.columns)
    total_profit = [100]
    for i in range(len(profit)):
        month_profit = 0
        for a in top5[i]:
            # 데이터 자르고 나서는, 같은 인덱스면 price 가 momentum보다 한 달 뒤므로 i-1
            if math.isnan(price.iloc[i-1][a]):
                month_profit += 0.01 / 5
                port_weight.iloc[i][a] = 0.01 / 5
                print('delisting')
            else:
                month_profit += profit.iloc[i][a] / 5
                port_weight.iloc[i][a] = profit.iloc[i][a] / 5
        # 원래 코드랑 다른 점
        total_profit.append(month_profit*total_profit[-1])
    result = total_profit[-1] / total_profit[0] - 1
    return result, total_profit, port_weight

"""
portfolio2 : rebalance top5 every December
"""
def port2(price, profit, top5, momentum):
    top5_df = pd.DataFrame(top5, index=momentum.index[:-1])
    top5_df = top5_df[top5_df.index.month==12]
    top5_li = top5_df.values.tolist()

    asset_year = 0
    denominator = 5
    total_profit = [100]
    collapse_sign = False
    asset_result = []
    month_record = []
    delisting = []
    port_weight = pd.DataFrame(np.zeros((len(profit), len(profit.columns))), index=profit.index, columns=profit.columns)

    for i in range(len(profit)):
        month_profit = 0
        assets = top5_li[asset_year]

        for a in assets:
            if math.isnan(price.iloc[i-1][a]):
                month_profit += 0.01 / denominator
                port_weight.iloc[i][a] = 0.01 / denominator
                # top5_li[asset_year].remove(a)
                delisting.append(a)
                collapse_sign = True
                print('delisting')
            else:
                month_profit += profit.iloc[i][a] / denominator
                port_weight.iloc[i][a] = profit.iloc[i][a] / 5
        # 상장폐지 있으면 삭제
        while delisting:
            top5_li[asset_year].remove(delisting.pop())

        total_profit.append(month_profit*total_profit[-1])
        if collapse_sign:
            denominator = len(assets)

        if i > 1 and (i+1) % 12 == 0:
            asset_year += 1
            denominator = 5
            collapse_sign = False
        asset_result.append(assets)
        result = total_profit[-1] / total_profit[0] - 1
    return result, total_profit, port_weight





# 실행부
directory = 'C://data_minsung/finance/Qraft/Required/Price.csv'
price, profit, top5, momentum = data_pre(directory)

# 수익률 158%
result, total_profit, port_weight = port1(price, profit, top5)
# port_weight.to_csv("C://data_minsung/finance/Qraft/result/port1.csv")
# plt.plot(total_profit)

# 수익률 126%
# result, total_profit, port_weight = port2(price, profit, top5, momentum)
# port_weight.to_csv("C://data_minsung/finance/Qraft/result/port2.csv")
# plt.plot(total_profit)

