import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# momentum = p(t-1) / p(t-12) - 1
"""
portfolio1 : rebalance top5 every month
"""
price = pd.read_csv('C://data_minsung/finance/Qraft/Required/Price.csv')
price = price.set_index('Unnamed: 0')
price.index = pd.to_datetime(price.index, format='%Y-%m-%d')

price.info()

# null value : B=46, N=14, P=97
# null인 경우 상장폐지. 결측치 아님
price.isnull().sum()

obs, cols = np.shape(price)

# momentum score 구하기
momentum = pd.DataFrame(np.zeros((obs, cols)), index=price.index, columns=price.columns)
for col in price:
    for i in range(12, obs):
        momentum[col][i] = price[col][i-1] / price[col][i-12] - 12
momentum = momentum.iloc[12:]

top5 = []
# 마지막 모멘텀은 필요 x - 모멘텀 통한 투자 다음달에 수익률 도출
for i in range(len(momentum)-1):
    asset = momentum.iloc[i].sort_values(ascending=False)[:5]
    top5.append(asset.index.values)

# price = price.iloc[12:]
# momentum 시작 81년 12월 31일, profit 시작 82년 1월 31일
profit = price.pct_change()[13:]+1
price = price.iloc[13:]

# len(price), len(top5), len(profit)

def port1(top5, profit, price):
    total_profit = [100]
    for i in range(len(profit)):
        month_profit = 0
        for a in top5[0]:
            if math.isnan(price.iloc[i][a]):
                print(i + 1, asset, profit.iloc[i + 1][asset])
                month_profit += 0.01 / 5
            else:
                month_profit += profit.iloc[i][a] / 5
        # 원래 코드랑 다른 점
        total_profit.append(month_profit*total_profit[-1])

    result = total_profit[-1] / total_profit[0] - 1
    return result

"""
portfolio2 : rebalance top5 every December
"""