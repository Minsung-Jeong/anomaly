import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

os.chdir('C://data_minsung/finance/Qraft')
etfs = pd.read_csv('./indexPortfolio/etfs.csv').set_index('Date')
macros = pd.read_csv('./indexPortfolio/macros.csv').set_index('Unnamed: 0')




"""
1.etfs가 시계열 데이터 특성을 가졌는지 확인
2. 보간법, 이상치제거, 수치변환, 다중공선성 제거, feature engineering 등
"""
# etfs:육안으로 보아도 추세나, 계절성 내포, white noise와 거리가 먼 모양
# 차분(difference)화 하기
etfs.plot()
etfs_ret = etfs.pct_change()
etfs_ret.index = pd.to_datetime(etfs_ret.index)

macros_ret = macros.pct_change()
macros_ret.index = pd.to_datetime(macros_ret.index)

st_date = macros_ret.index[0]
etfs_ret = etfs_ret.apply(lambda x: x[st_date:])

etfs_ret.plot()

#보간법 : nan값은 이전과 변동 없는 것으로 간주하여 0.0으로 결측치 채움
macros_ret = macros_ret.fillna(0.0)
etfs_ret = etfs_ret.fillna(0.0)

# 상관계수 heatmap 통해 대략적인 파악
cmap = sns.light_palette("darkgray", as_cmap=True)
sns.heatmap(etfs_ret.corr(), annot=True, cmap=cmap)
plt.show()

sns.heatmap(macros_ret.corr(), annot=True, cmap=cmap)
plt.show()

# 다중공선성 제거 : VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(etfs_ret.values, i) for i in range(etfs_ret.shape[1])]
vif.index = etfs_ret.columns


# 월별 인덱스
add_ = etfs_ret.index[0]
last_idx = etfs_ret.resample('M').last().index
first_idx = last_idx + timedelta(days=1)
first_idx = list(first_idx)[:-1]
first_idx.insert(0, add_)

first_idx[:10]
last_idx[:10]
