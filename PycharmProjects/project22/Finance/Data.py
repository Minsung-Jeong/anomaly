import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from datetime import datetime, timedelta
import math
import quantstats as qs
import matplotlib.pyplot as plt

os.getcwd()
# os.chdir("C://data_minsung/finance")
#
# # 소비자 물가지수
# cpi = pd.read_csv('./ConsumPIAUCSL_M.csv')
# col = cpi.columns
# cpi_change = cpi.CPIAUCSL.pct_change()
# plt.plot(cpi.DATE, cpi_change)

# pd_datareader
start_day = datetime(2008,1,1) # 시작일
end_day = datetime(2021,6,12) # 종료일
SPY = pdr.get_data_yahoo('SPY', start_day - timedelta(days=365), end_day)['Adj Close']


# VAA 대상 자산 값 불러오기

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

profit_col_list = [col+'_P' for col in df_RCU[RU+CU].columns]
df_RCU[profit_col_list] = df_RCU[RU+CU].pct_change()


plt.plot(df_RCU[profit_col_list].iloc[:,1].resample(rule='M').last())