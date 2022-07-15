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
포트폴리오 1:
12M-1M 상위 5개를 매달 동일비중 리밸런싱
score = (1개월전/12개월전 - 1)
price 미존재는 종목 상장폐지 : 수익률 -99%로 대체, 해당 종목 비중 다른 종목의 비중에 비례해 분배
다른 종목 비중 비례 분배는 매달 리밸런싱할 때는 불필요
"""
def get_momentum(x):
    temp_list = np.zeros(len(x.index))
    momentum = pd.Series(temp_list, index=x.index)
    try:
        # timedelta 월별 설정이 불가
        before1 = price[x.name - timedelta(days=35) : x.name - timedelta(days=30)].iloc[-1]
        before12 = price[x.name - timedelta(days=370) : x.name - timedelta(days=365)].iloc[-1]
        momentum = before1/before12 - 1
    except Exception as e:
        # print("Error : ", str(e))
        pass
    return momentum

# 앞선 12개월은 모멘텀 스코어 없으므로 생략
m_score = price.apply(lambda x: get_momentum(x), axis=1).iloc[12:]
price = price.iloc[12:]

Asset = []
for i in range(len(m_score)):
    top5 = m_score.iloc[i].sort_values(ascending=False)[:5]
    Asset.append(top5.index.values)

Allo_Asset = pd.DataFrame(Asset, index=price.index)

profit_col = [x+'_P' for x in price.columns]
price[profit_col] = price.pct_change()

# 포트폴리오 수익률 도출
price['PROFIT'] = 0
# 배정하는 for문 안에서 조건문 추가를 통해서 투자-폐지 하는 로직 짜기
pr_idx = price.index
for i in range(len(price)):
    top5 = Allo_Asset.values[i]
    profit = 0
    # 왜 첫번째는 생략하더라?
    if i != 0:
        for asset in top5:
            # 5개의 자산 균등분배
            # 자산값이 Nan이면 -0.99의 수익률 배정
            if math.isnan(price.iloc[i][asset]):
                # 에러나는 부분 - 당연히 값도 할당이 안되고 있음
                price.loc[pr_idx[i],asset+ "_P"] = -0.99
                print('상장폐지 {0}자산, {1}번째'.format(asset, i))
            profit += price[profit_col].iloc[i][asset + "_P"]*(1/5)
    price.loc[price.index[i], 'PROFIT'] = profit

plt.plot(price['PROFIT'])
plt.plot(pd.DataFrame(np.zeros(len(price)), index=price.index))