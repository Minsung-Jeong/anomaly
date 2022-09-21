import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
os.chdir('C://data_minsung/finance/Qraft')

etfs = pd.read_csv('./indexPortfolio/etfs.csv').set_index('Date')
etfs.index = pd.to_datetime(etfs.index, format='%Y-%m-%d')
etfs_ret = etfs.pct_change().iloc[1:]

# etfs.index[0].strftime('%d')

last_d = etfs_ret.resample('M').last().index
first_d = last_d[:-1] + timedelta(days=1)
first_d.insert(0, etfs.index[0])


# macro데이터, etf 가격 데이터
# 1.횡적 리스크 모델 활용해 포폴 비중으로 변환(Sharpe Maximized)
# 2.neural net : label = 비중, x = macros, etfs

