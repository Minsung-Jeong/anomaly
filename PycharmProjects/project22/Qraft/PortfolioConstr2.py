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

m_score = price.apply(lambda x: get_momentum(x), axis=1)
profit_col = [x+'_P' for x in price.columns]
asset_profit = price[price.columns].pct_change()
price[profit_col] = asset_profit

Asset = []
for i in range(len(m_score)):
    top5 = m_score.iloc[i].sort_values(ascending=False)[:5]
    Asset.append(top5.index.values)

# 모멘텀 스코어는 12개월 지난 시점부터 생성
Asset = pd.DataFrame(Asset, index=price.index).iloc[12:]
price = price.iloc[12:]
# 매년 12월의 상위 자산
Asset = Asset[Asset.index.month == 12]
Asset_li = [np.array(x) for x in Asset.values]
# 포트폴리오 수익률 도출
price['PROFIT'] = 0
a_year = 0 # 매년 12월로 이뤄진 Asset의 index 잡아줌
portfolio = []
# 모멘텀 상위 자산 배분통한 수익률 도출 + 상장폐지 -99% 적용
for i in range(len(price)):
    p_date = price.index[i]
    a_date = Asset.index[a_year]
    top5 = Asset_li[a_year]
    profit = 0
    portfolio.append(top5)
    if (p_date.year == a_date.year and p_date.month == a_date.month) or (p_date.year == a_date.year+1):
        for asset in top5:
            if math.isnan(price.iloc[i][asset]):
                price.loc[price.index[i], asset + "_P"] = -0.99
                print('상장폐지 {0}자산, {1}번째 달 수익률 -.99'.format(asset, i))
                profit += price[profit_col].iloc[i][asset + "_P"] * (1 / len(top5))
                Asset_li[a_year] = np.delete(top5, np.where(top5 == asset))
                # top5 = np.delete(top5, np.where(top5 == asset))
                # Asset_li[a_year] = top5
            else:
                profit += price[profit_col].iloc[i][asset + "_P"] * (1 / len(top5))
        price.loc[price.index[i], 'PROFIT'] = profit
    # 다음 해 12월이 되면 asset의 인덱스 증가
    if(p_date.year == a_date.year + 1 and p_date.month == a_date.month):
        a_year += 1

plt.plot(price['PROFIT'])
plt.plot(pd.DataFrame(np.zeros(len(price)), index=price.index))

# os.chdir('C://data_minsung/finance/Qraft/Required')
# pd.DataFrame(portfolio, index=price.index).to_csv('./Result/Construct2.csv')


#  portfolio 평가
os.chdir("C://data_minsung/finance/Qraft")

# US500 data 로 미국 시장(S&P500) 관한 정보
market = pd.read_csv("./US500_CASH.csv")['Price'].values[::-1]
market_profit = pd.DataFrame(market).pct_change().values[2:]
market = pd.DataFrame(market[2:], index=price.index)

# 미국 2년 채권수익률 : risk free
bill = pd.read_csv("./DGS2.csv").values[2:,1]

# 포트폴리오, 시장 누적 수익률, 무위험자산 수익률
plus1 = price['PROFIT']+1
port_return = (plus1.cumprod()-1)[-1]

m_plus1 = market_profit+1
market_return = (m_plus1.cumprod()-1)[-1]

# 이자율이므로 12 나누고, 백분율 고려해 100나누기 => 1200
b_plus1 = bill/1200 + 1
bill_return = (b_plus1.cumprod()-1)[-1]


def get_beta(profit, market):
    market = np.reshape(market, (len(market),))
    covar = np.cov(np.stack((profit, market), axis=0))
    var = np.var(market)
    beta = covar[0,1]/var
    return beta

port_down = []
for x in price['PROFIT'].values:
    if x < 0:
        port_down.append(x)

alpha = port_return - market_return
beta = get_beta(price['PROFIT'].values, market_profit)
# 투자자 부담 리스크를 자산 수익률이 얼마나 잘 보상하는지, 높을 수록 좋다
# 샤프와 소르티노에 대해서도 좀 봐야하나
sharpe = (port_return - bill_return) / np.std(price['PROFIT'])
sortino = (port_return - bill_return) / np.std(port_down)
