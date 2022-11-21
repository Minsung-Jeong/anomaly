import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 목표 : 이상적인 sharpe Maximized 포트폴리오 구축 및 결과분석

os.chdir('C://data_minsung/finance/Qraft')

etfs = pd.read_csv('./indexPortfolio/etfs.csv').set_index('Date')
etfs.index = pd.to_datetime(etfs.index)

# EMB 결측치가 3413으로 가장 많다
etfs.isna().sum()

# 12월 31일
# temp_dec = etfs[etfs.index.month == 12]
# yearly_dec = temp_dec[temp_dec.index.day==31]

# # 1월 1일
# temp = etfs[etfs.index.month == 1]
# yearly_jan = temp[temp.index.day == 1]
# yearly_ret = yearly_jan.pct_change()
#
# temp = etfs[etfs.index.year==1994]
#
etfs_ret = etfs.pct_change().dropna()

monthly = etfs_ret.resample('M').last()

year_idx = [monthly.index[12*i] for i in range(len(monthly)//12+1)]
yearly = pd.DataFrame(np.zeros((len(monthly)//12+1 , len(monthly.columns))), index=year_idx, columns=monthly.columns)

for i in range(len(monthly)//12):
    yearly.iloc[i] = monthly.iloc[(i-1)*12:i*12].mean()
yearly.dropna(inplace=True)

# yearly : 94/2/28~21/2/28 , observation 28개
# montly : 93/2/28~21/5/31, observation 340개

port_ret = []
port_risk = []
port_weights = []
sharpe_ratio = []

stock = etfs_ret.columns.values

for _ in range(100):

    # 임의의 포트 비중 생성
    weights = np.random.random(len(stock))
    weights /= np.sum(weights)

    returns  = np.dot(weights, a)