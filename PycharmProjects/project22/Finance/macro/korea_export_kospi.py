import pandas as pd
import os
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression
import numpy as np

os.chdir('C://data_minsung')


# export 데이터 전처리----------------------------------------------------
export_df = pd.read_csv("./finance/macro/korea_trade.csv")
export_df['date'] = [str(export_df['date'][i]).replace('.','-') for i in range(len(export_df))]
export_df['date'] = pd.to_datetime(export_df['date'])

#index= 9, 21, 30 ...., 에서 10월이 1월로 기록된 에러 해결
for i in range(len(export_df)//12+1):
    export_df['date'][[i*12+9]] = export_df['date'][i*12+9] + relativedelta(months=9)
    print(export_df['date'][[i*12+9]])

export_df.set_index('date', inplace=True)

# 숫자에 ',' 제거한 뒤 int로 변경
for i in range(len(export_df)):
    for j in range(len(export_df.iloc[i])):
        export_df.iloc[i][j] = int(export_df.iloc[i][j].replace(',',''))

# --------------------------------------------------------------------------
# kospi 전처리

kospi_df = pd.read_csv("./finance/macro/kospi_pro.csv")
kospi_df['date'] = [str(kospi_df['date'][i]).replace('.','-') for i in range(len(kospi_df))]
kospi_df['date'] = pd.to_datetime(kospi_df['date'])

# index = 8, 20, 32 ...에서 10월이 1월로 나오는 에러
for i in range(len(kospi_df)//12+1):
    kospi_df['date'][[i*12+8]] = kospi_df['date'][i*12+8] + relativedelta(months=9)

    print(kospi_df['date'][[i*12+8]])

kospi_df.set_index('date', inplace=True)

# ---------------회귀분석
from statsmodels.formula.api import ols
from sklearn.preprocessing import MinMaxScaler
# export 1월로 시작, kospi 2월로 시작
export_df.index[1]
kospi_df.index[0]

total = kospi_df
total['X'] = export_df['export_total'].values[:-1].copy()
total.columns

scaler = MinMaxScaler()
total[:] = scaler.fit_transform(total[:])

reg = ols('value~X',data=total).fit()
reg.summary()