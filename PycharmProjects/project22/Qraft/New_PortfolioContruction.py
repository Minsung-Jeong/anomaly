import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import math
import matplotlib.pyplot as plt

# 12-1모멘텀 = 1개월전/12개월전-1
# 1번 - 매달 동일비중 리밸런싱
# 2번 - 매년 12월 동일비중 리밸런싱

# 상위 5개 종목을 동일비중으로 구성,
# Nan 상폐, 상폐시 수익률 -99%, 해당종목 비중을 다른 종목의 비중에 비례해 배분

# 21개 가격지표, 469개 관측치
price = pd.read_csv('C://data_minsung/finance/Qraft/Required/Price.csv')
price = price.set_index('Unnamed: 0')
price.index = pd.to_datetime(price.index, format='%Y-%m-%d')

# B=46, N=14, P=97
price.isnull().sum()

# momentum = pd.Series(np.zeros(len(price.index)), index=price.index)
momentum = pd.DataFrame(np.zeros((len(price.index), len(price.columns))), index=price.index, columns=price.columns)
for x in price.columns:
    for i in range(len(price.index)):
        if i >= 12:
            momentum[x][i] = price[x][i-1]/price[x][i-12] - 1

top5 = []
for i in range(12, len(momentum.index)):
    asset = momentum.iloc[i].sort_values(ascending=False)[:5]
    top5.append(asset.index.values)

top5 = pd.DataFrame(top5, index=price.index[12:])
profit = price.pct_change()[12:]
price = price.iloc[12:]
momentum = momentum.iloc[12:]

# profit = 0, momentum = 0 or nan이면 상장폐지
# 7월 모멘텀 스코어 통해 8월 수익률 도출
def m_port(top5, profit, price):
    total_profit = [100]
    # 모멘텀 스코어는 i, profit은 i+1로 인덱싱 -> 모멘텀 스코어로 결정한 결과는 다음 달이므로
    for i in range(len(profit)-1):
        assets = top5.iloc[i]
        month_profit = 0
        for asset in assets:
            if math.isnan(price.iloc[i][asset]):
                print(i+1, asset, profit.iloc[i+1][asset])
                month_profit += -0.99
            else:
                month_profit += profit.iloc[i+1][asset]
        total_profit.append((month_profit/5 + 1)*total_profit[-1])
    return total_profit



def y_port(top5, profit, price):
    year_top5 = top5[top5.index.month == 12]
    year_top5_li = year_top5.values.tolist()
    asset_year = 0

    denominator = 5
    total_profit = [100]
    collapse_sign = False #상폐 발생 시 True
    asset_result = []
    for i in range(1, len(profit)):
        assets = year_top5_li[asset_year]
        month_profit = 0
        for asset in assets:
            if math.isnan(price.iloc[i][asset]):
                month_profit += -0.99 / denominator
                year_top5_li[asset_year].remove(asset)
                collapse_sign = True
            else:
                month_profit += profit.iloc[i][asset] / denominator
        total_profit.append((month_profit + 1) * total_profit[-1])
        # 분모의 변동은 for ~assets 구문 밖에서 이뤄져야
        if collapse_sign:
            denominator = len(assets)
        # 12개월이 지남에 따라 스코어 인덱스(연도별) 증가
        if i > 1 and i % 12 == 0:
            asset_year += 1
            # 연도 변경에 따라 새로운 자산군 -> 분모, 상폐여부 초기화
            denominator = 5
            collapse_sign = False
        asset_result.append(assets)
    return total_profit, asset_result

monthly_profit = m_port(top5, profit, price)
yearly_profit, yearly_asset = y_port(top5, profit, price)


plt.plot(yearly_profit)
plt.plot(monthly_profit)