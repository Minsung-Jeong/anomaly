import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import math
# 자산 종류 21개
price = pd.read_csv('C://data_minsung/finance/Qraft/Required/Price.csv')
price = price.set_index('Unnamed: 0')
price_col = price.columns
price.index = pd.to_datetime(price.index, format='%Y-%m-%d')
price.index.name = 'Date'


# 결측치 B:46개, N:14개, P:97개
price.isnull().sum()
"""
포트폴리오 1:
12M-1M 상위 5개를 매달 동일비중 리밸런싱
score = (1개월전/12개월전 - 1)
price 미존재는 종목 상장폐지 : 수익률 -99%로 대체, 해당 종목 비중 다른 종목의 비중에 비례해 분배
"""


# 이 구조 안 이쁜데...좀 더 이쁜 방법을 강구
# price['A']['1981-10-31']
# price['A'][price.iloc[10].name- timedelta(weeks=5):price.iloc[10].name- timedelta(weeks=4)]


def get_momentum(x):
    temp_list = np.zeros(len(x.index))
    momentum = pd.Series(temp_list, index=x.index)
    try:
        # print(x.name)
        before1 = price[x.name - timedelta(days=35) : x.name - timedelta(days=30)].iloc[-1]
        before12 = price[x.name - timedelta(days=370) : x.name - timedelta(days=365)].iloc[-1]
        # print(before1)
        # print(before12)
        momentum = before1/before12 - 1
    except Exception as e:
        # print("Error : ", str(e))
        pass
    return momentum

m_score = price.apply(lambda x: get_momentum(x), axis=1)

# m_col = [x+'_M' for x in price.columns]

# 앞선 12개월은 모멘텀 스코어 없으므로 생략
m_score = m_score[12:]
price = price.iloc[12:]


Asset = []
for i in range(len(m_score)):
    top5 = m_score.iloc[i].sort_values(ascending=False)[:5]
    Asset.append(top5.index.values)

Allo_Asset = pd.DataFrame(Asset, index=price.index)

profit_col = [x+'_P' for x in price.columns]
price[profit_col] = price.pct_change()

# price nan이
# 특정 월의 top5 뽑기
temp_5 = Allo_Asset.values[-46]

# 뽑은 top5 중에서 nan 찾고 action 취하기
for asset in temp_5:
    # nan인 경우를 찾기, 찾고
    if math.isnan(price.iloc[-46][asset]):
        price.iloc[-46][asset+ "_P"] == -0.99
        print('상장폐지 이슈로 인한 수익률 -.99 ', 'idx, asset')

type(price.iloc[-46][asset])
# profit = 0
# for asset in temp_5:
#     profit+=price[profit_col].iloc[1][asset + "_P"]
#     print(price[profit_col].iloc[1][asset + "_P"])
#     print("profit", profit)




# 여기에다가 상장폐지 예외처리하면 됨
price['PROFIT'] = 0
# 배정하는 for문 안에서 조건문 추가를 통해서 투자-폐지 하는 로직 짜기
for i in range(len(price)):
    top5 = Allo_Asset.values[i]
    profit = 0

    if i != 0:
        for asset in top5:
            # 5개의 자산 균등분배
            if math.isnan(price.iloc[i][asset]):
                # 에러나는 부분 - 당연히 값도 할당이 안되고 있음
                price.iloc[i][asset+ "_P"] = -0.99
                print('상장폐지 {0}자산, {1}번째'.format(asset, i))

            profit += price[profit_col].iloc[i][asset + "_P"]*(1/5)

    price.loc[price.index[i], 'PROFIT'] = profit
    # breakpoint()
price[profit_col].iloc[360]
price['PROFIT'].iloc[360]
price.iloc[360]["P_P"]

"""
포트폴리오 2:
12-1M 상위 5개를 매년 12월에 동일비중 리밸런싱
"""