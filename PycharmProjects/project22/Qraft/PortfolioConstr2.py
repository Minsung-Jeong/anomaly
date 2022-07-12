import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import math
import matplotlib.pyplot as plt

# 자산 종류 21개
price = pd.read_csv('C://data_minsung/finance/Qraft/Required/Price.csv')
price = price.set_index('Unnamed: 0')
price.index = pd.to_datetime(price.index, format='%Y-%m-%d')
price.index.name = 'Date'

# 결측치 B:46개, N:14개, P:97개
price.isnull().sum()

"""
포트폴리오2
12-1M 상위 5개를 매년 12월에 동일비중 리밸런싱
상폐시 : -99%수익 + 해당 종목 비중 다른 종목의 비중에 비례해 분배
"""

def get_momentum(x):
    temp_list = np.zeros(len(x.index))
    momentum = pd.Series(temp_list, index=x.index)
    try:
        # timedelta 월별 설정 불가
        before1 = price[x.name - timedelta(days=35) : x.name - timedelta(days=30)].iloc[-1]
        before12 = price[x.name - timedelta(days=370) : x.name - timedelta(days=365)].iloc[-1]
        momentum = before1/before12 - 1
    except Exception as e:
        # print("Error : ", str(e))
        pass
    return momentum

price_col = price.columns
momentum_col = [x+'_M' for x in price_col]
m_score = price.apply(lambda x: get_momentum(x), axis=1)

profit_col = [x+'_P' for x in price_col]
asset_profit = price[price_col].pct_change()
price[profit_col] = asset_profit

Asset = []
for i in range(len(m_score)):
    top5 = m_score.iloc[i].sort_values(ascending=False)[:5]
    Asset.append(top5.index.values)


# 모멘텀 스코어는 12개월 지난 시점부터 생성
Asset = pd.DataFrame(Asset, index=price.index).iloc[12:]
price = price.iloc[12:]

# 포트폴리오 수익률 도출
price['PROFIT'] = 0
pr_idx = price.index
broken = 0

# 매년 12월의 상위 자산
Asset = Asset[Asset.index.month == 12]
Asset_li = [list(x) for x in Asset.values]


a_year = 0 # 매년 12월로 이뤄진 Asset의 index 잡아줌
# 모멘텀 상위 자산 배분통한 수익률 도출 + 상장폐지 -99% 적용
# for i in range(len(price)):
for i in range(362):
    p_date = price.index[i]
    a_date = Asset.index[a_year]
    top5 = Asset_li[a_year]
    profit = 0
    if (p_date.year == a_date.year and p_date.month == a_date.month) or (p_date.year == a_date.year+1 and p_date.month != a_date.month):
        for asset in top5:
            # 5개의 자산 균등분배
            # 자산값==Nan, -0.99의 수익률 배정 + 자산배분배(해당자산 -99% 처리 후 나머지 자산 1/4)
            # 두 단계로 고려, 1단계 상폐일어날 경우 비중 줄이기, 2단계 추가상폐 때 처리

            # price=nan이 시작되면 상폐, 첫 nan이 나오면 그 순간 top5에서 해당 자산이 다 삭제
            # 조건문은 가격이 nan일 때 모두를 고려하게 된다.

            if math.isnan(price.iloc[i][asset]):
                # broken += 1
                price.loc[pr_idx[i], asset + "_P"] = -0.99
                print('상장폐지 {0}자산, {1}번째'.format(asset, i))

                profit += price[profit_col].iloc[i][asset + "_P"] * (1 / len(top5))
                # 여기서 np.where 로 인덱스를 못 당겨온다
                print(top5, asset)
                print('같냐?', top5 == 'P')
                print(np.where(top5 == asset))
                print(np.delete(top5, np.where(top5 == asset)))
                top5 = np.delete(top5, np.where(top5 == asset))
                Asset_li[a_year] = top5
                print('삭제후', top5, asset)
            else:
                profit += price[profit_col].iloc[i][asset + "_P"] * (1 / len(top5))
        price.loc[price.index[i], 'PROFIT'] = profit

    # 다음 년도로 넘어가는 로직
    elif(p_date.year == a_date.year + 1 and p_date.month == a_date.month):
        a_year += 1
        # broken = 0
    # print('--', top5, asset)


# temp_min = np.min(price['PROFIT'].values)
# temp_price = price['PROFIT'].values
# np.where(temp_price==temp_min)


plt.plot(price['PROFIT'])
plt.plot(pd.DataFrame(np.zeros(len(price)), index=price.index))

# temp = np.array([1,2,3,4,5,6,7,8,9])
# Asset_li = [temp, temp, temp]
#
# for x in Asset_li[0]:
#     print(x)
#     Asset_li[1] = np.delete(temp, 0)