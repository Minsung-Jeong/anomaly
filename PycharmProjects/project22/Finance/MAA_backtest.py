import pandas_datareader as pdr
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import math
import quantstats as qs
import numpy as np

# 예측 수익률 (+, -) 유무로 공격자산 방어자산 선택하고
# 각 자산군의 가장 높은 모멘텀 스코어를 뽑아서 투자 진행하기
# binary : 양수이면 1(공격), 음수이면 0(방어)으로 설정

decision = pred_average
# decision = RNN_pd
plt.plot(decision)
plt.show()
df_RCU = df_RCU.resample('M').last()[decision.index[0]:decision.index[-1]]
df_RCU['PRED'] = decision

# def select_asset(x):
#     asset = pd.Series([0,0], index=['ASSET', 'PRICE'])
#     if x['PRED'] > 0.5:
#         max_momentum = max(x['SPY_M'], x['VEA_M'], x['EEM_M'], x['AGG_M'])
#     else:
#         max_momentum = max(x['LQD_M'], x['SHY_M'], x['IEF_M'])
#     asset['ASSET'] = x[x==max_momentum].index[0][:3]
#     asset['PRICE'] = x[asset['ASSET']]
#     return asset

def select_asset_safe(x):
    asset = pd.Series([0,0], index=['ASSET', 'PRICE'])
    if x['PRED'] > 0.00 and x['SPY_M'] >0 and x['VEA_M'] > 0 and x['EEM_M'] >0 and x['AGG_M'] > 0:
        max_momentum = max(x['SPY_M'], x['VEA_M'], x['EEM_M'], x['AGG_M'])
    else:
        max_momentum = max(x['LQD_M'], x['SHY_M'], x['IEF_M'])
    asset['ASSET'] = x[x==max_momentum].index[0][:3]
    asset['PRICE'] = x[asset['ASSET']]
    return asset


df_RCU.iloc[0]
# 자산 선택 및 수익률
df_RCU[['ASSET', 'PRICE']] = df_RCU.apply(lambda x: select_asset_safe(x), axis=1)
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

plt.plot(df_RCU['LOG_PROFIT'])
# df_RCU['LOG_PROFIT'].to_csv('./MAA_log_profit.csv')
qs.reports.basic(df_RCU['PROFIT']/100)