import pandas_datareader as pdr
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import math
import quantstats as qs
import numpy as np
from sklearn import preprocessing

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
        breakpoint()
        before1 = df_RCU[x.name - timedelta(days=35) : x.name - timedelta(days=30)].iloc[-1][RU + CU]
        before3 = df_RCU[x.name - timedelta(days=95) : x.name - timedelta(days=90)].iloc[-1][RU + CU]
        before6 = df_RCU[x.name - timedelta(days=185) : x.name - timedelta(days=180)].iloc[-1][RU + CU]
        before12 = df_RCU[x.name - timedelta(days=370) : x.name - timedelta(days=365)].iloc[-1][RU + CU]
        momentum = 12 * (x / before1 - 1) + 4 * (x / before3 - 1) + 2 * (x / before6 - 1) + (x / before12 - 1)
    except Exception as e:
        # print("Error : ", str(e))
        pass
    return momentum


mom_col_list = [col+'_M' for col in df_RCU[RU+CU].columns]
df_RCU[mom_col_list] = df_RCU[RU+CU].apply(lambda x: get_momentum(x), axis=1)

df_RCU.head()




# 자산 수익률의 평균을 통해 y값 도출(정규화 유무에 따라 2개의 y)
profit_col_list = [col+'_P' for col in df_RCU[RU+CU].columns]
df_RCU[profit_col_list] = df_RCU[RU+CU].pct_change()
df_RCU[profit_col_list] = df_RCU[profit_col_list].fillna(0)

scaler = preprocessing.StandardScaler().fit(df_RCU[profit_col_list])
profit_normalize = scaler.transform(df_RCU[profit_col_list])

y1 = df_RCU[profit_col_list].mean(axis=1)
y2 = pd.DataFrame(profit_normalize, index=y1.index)
y2 = y2.mean(axis=1)

np.abs(y1).mean()
y1.std()

np.abs(y2).mean()
y2.std()


y1.max()
y2.max()
plt.plot(y2)
plt.show()