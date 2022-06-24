import pandas_datareader as pdr
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import math
import quantstats as qs
import numpy as np


# pandas 설정 및 메타데이터 세팅
pd.options.display.float_format = '{:.4f}'.format
pd.set_option('display.max_columns', None)

start_day = datetime(2008,1,1) # 시작일
end_day = datetime(2022,6,12) # 종료일

# RU : Risky Universe
# CU : Cash Universe
# BU : Benchmark Universe
RU = ['SPY','VEA','EEM','AGG'] #미국, 선진국, 개도국, 미국총채권
CU = ['LQD','SHY','IEF'] #미국회사채, 미국단기국채, 미국 중기국채
BU = ['^GSPC','^IXIC','^KS11','^KQ11'] # S&P500 지수, 나스닥 지수, 코스피 지수, 코스닥 지수


def get_price_data(RU, CU, BU):
    df_RCU = pd.DataFrame(columns=RU + CU)
    df_BU = pd.DataFrame(columns=BU)

    for ticker in RU + CU:
        df_RCU[ticker] = pdr.get_data_yahoo(ticker, start_day - timedelta(days=365), end_day)['Adj Close']

    for ticker in BU:
        df_BU[ticker] = pdr.get_data_yahoo(ticker, start_day - timedelta(days=365), end_day)['Adj Close']

    return df_RCU, df_BU


df_RCU, df_BU = get_price_data(RU, CU, BU)


# 모멘텀 스코어 뽑는 함수
def get_momentum(x):
    temp_list = [0 for i in range(len(x.index))]
    momentum = pd.Series(temp_list, index=x.index)
    try:
        # print(x.name)
        # print(x)
        # breakpoint()
        before1 = df_RCU[x.name - timedelta(days=35) : x.name - timedelta(days=30)].iloc[-1][RU + CU]
        before3 = df_RCU[x.name - timedelta(days=95) : x.name - timedelta(days=90)].iloc[-1][RU + CU]
        before6 = df_RCU[x.name - timedelta(days=185) : x.name - timedelta(days=180)].iloc[-1][RU + CU]
        before12 = df_RCU[x.name - timedelta(days=370) : x.name - timedelta(days=365)].iloc[-1][RU + CU]
        momentum = 12 * (x / before1 - 1) + 4 * (x / before3 - 1) + 2 * (x / before6 - 1) + (x / before12 - 1)
    except Exception as e:
        # print("Error : ", str(e))
        pass
    return momentum


# def get_momentum(x):
#     temp_list = [0 for i in range(len(x.index))]
#     momentum = pd.Series(temp_list, index=x.index)
#     try:
#         # print(x.name)
#         # print(x)
#         # breakpoint()
#         before1 = df_RCU[x.name - timedelta(days=35) : x.name - timedelta(days=30)].iloc[-1][RU + CU]
#         before2 = df_RCU[x.name - timedelta(days=65): x.name - timedelta(days=60)].iloc[-1][RU + CU]
#         before3 = df_RCU[x.name - timedelta(days=95) : x.name - timedelta(days=90)].iloc[-1][RU + CU]
#         before4 = df_RCU[x.name - timedelta(days=125): x.name - timedelta(days=120)].iloc[-1][RU + CU]
#         before5 = df_RCU[x.name - timedelta(days=155): x.name - timedelta(days=150)].iloc[-1][RU + CU]
#         before6 = df_RCU[x.name - timedelta(days=185) : x.name - timedelta(days=180)].iloc[-1][RU + CU]
#         before7 = df_RCU[x.name - timedelta(days=215): x.name - timedelta(days=210)].iloc[-1][RU + CU]
#         before8 = df_RCU[x.name - timedelta(days=245) : x.name - timedelta(days=240)].iloc[-1][RU + CU]
#         before9 = df_RCU[x.name - timedelta(days=275): x.name - timedelta(days=270)].iloc[-1][RU + CU]
#         before10 = df_RCU[x.name - timedelta(days=305) : x.name - timedelta(days=300)].iloc[-1][RU + CU]
#         before11 = df_RCU[x.name - timedelta(days=335) : x.name - timedelta(days=330)].iloc[-1][RU + CU]
#
#         before12 = df_RCU[x.name - timedelta(days=370) : x.name - timedelta(days=365)].iloc[-1][RU + CU]
#
#         momentum = 12*(x/before1-1)+11*(x/before2-1)+10*(x/before3-1)+9*(x/before4-1)+8*(x/before5-1)+7*(x/before6-1)+6*(x/before7-1)+5*(x/before8-1)+4*(x/before9-1)+3*(x/before10-1)+2*(x/before11-1)+(x/before12-1)
#     except Exception as e:
#         # print("Error : ", str(e))
#         pass
#     return momentum

mom_col_list = [col+'_M' for col in df_RCU[RU+CU].columns]
df_RCU[mom_col_list] = df_RCU[RU+CU].apply(lambda x: get_momentum(x), axis=1)

df_RCU.head()

# 백테스트 대상 기간 데이터 추출
df_RCU = df_RCU[start_day:end_day]

# 매월 말일 데이터만 추출(리밸런싱에 사용), first/last 가능
df_RCU_m = df_RCU.resample(rule='M').last()

# last로 하면 월말 데이터가 없을 때 그 이전 날이나 제일 늦은 날로 채워줌
df_RCU.iloc[-577]==df_RCU_m.iloc[-29]

#  VAA 전략에 맞춘 자산선택
def select_asset(x):
    asset = pd.Series([0,0], index=['ASSET', 'PRICE'])

    if x['SPY_M'] >0 and x['VEA_M'] > 0 and x['EEM_M'] >0 and x['AGG_M'] > 0:
        max_momentum = max(x['SPY_M'], x['VEA_M'], x['EEM_M'], x['AGG_M'])

    # 공격 자산 중 하나라도 0이하면 방어자산 중 최고 모멘텀 선정
    else:
        max_momentum = max(x['LQD_M'], x['SHY_M'], x['IEF_M'])

    asset['ASSET'] = x[x==max_momentum].index[0][:3]
    asset['PRICE'] = x[asset['ASSET']]
    return asset


df_RCU[['ASSET','PRICE']] = df_RCU.apply(lambda x: select_asset(x), axis=1)
df_RCU.tail()

# 각 자산별 수익률 계산
profit_col_list = [col+'_P' for col in df_RCU[RU+CU].columns]
df_RCU[profit_col_list] = df_RCU[RU+CU].pct_change()


# 매월 수익률 & 누적 수익률 계산
df_RCU['PROFIT'] = 0
df_RCU['PROFIT_ACC'] = 0
df_RCU['LOG_PROFIT'] = 0
df_RCU['LOG_PROFIT_ACC'] = 0

for i in range(len(df_RCU)):
    profit = 0
    log_profit = 0

    if i != 0:
        profit = df_RCU[df_RCU.iloc[i - 1]['ASSET'] + '_P'].iloc[i]
        log_profit = math.log(profit + 1)

    df_RCU.loc[df_RCU.index[i], 'PROFIT'] = profit
    df_RCU.loc[df_RCU.index[i], 'PROFIT_ACC'] = (1 + df_RCU.loc[df_RCU.index[i - 1], 'PROFIT_ACC']) * (1 + profit) - 1
    df_RCU.loc[df_RCU.index[i], 'LOG_PROFIT'] = log_profit
    df_RCU.loc[df_RCU.index[i], 'LOG_PROFIT_ACC'] = df_RCU.loc[df_RCU.index[i - 1], 'LOG_PROFIT_ACC'] + log_profit

# 백분율을 %로 표기
df_RCU[['PROFIT', 'PROFIT_ACC', 'LOG_PROFIT', 'LOG_PROFIT_ACC']] = df_RCU[['PROFIT', 'PROFIT_ACC', 'LOG_PROFIT',
                                                                           'LOG_PROFIT_ACC']] * 100
df_RCU[profit_col_list] = df_RCU[profit_col_list] * 100


qs.reports.basic(df_RCU['PROFIT']/100)
# qs.reports.plots(df_RCU['PROFIT']/100)


plt.plot(df_RCU['PROFIT_ACC'])

# temp = df_RCU['PROFIT']
# temp.iloc[0]
#
# for i in range(len(temp)):
#     if i > 0:
#         temp.iloc[i] = temp.iloc[i-1] + temp.iloc[i]
#
# plt.plot(temp)