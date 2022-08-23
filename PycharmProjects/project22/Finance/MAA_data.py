import pandas_datareader as pdr
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import math
import quantstats as qs
import numpy as np
from sklearn import preprocessing
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression


# pandas 설정 및 메타데이터 세팅
pd.options.display.float_format = '{:.4f}'.format
pd.set_option('display.max_columns', None)

start_day = datetime(2008,1,1) # 시작일
end_day = datetime(2022,6,12) # 종료일

os.chdir("C://data_minsung")

# # RU : Risky Universe
# # CU : Cash Universe
# # BU : Benchmark Universe
RU = ['SPY','VEA','EEM','AGG'] #미국, 선진국, 개도국, 미국총채권
CU = ['LQD','SHY','IEF'] #미국회사채, 미국단기국채, 미국 중기국채
BU = ['^GSPC','^IXIC','^KS11','^KQ11'] # S&P500 지수, 나스닥 지수, 코스피 지수, 코스닥 지수
CA = ['VWO', 'BND']

def get_price_data(RU, CU, BU, CA):
    df_RCU = pd.DataFrame(columns=RU + CU)
    df_BU = pd.DataFrame(columns=BU)
    df_CA = pd.DataFrame(columns=CA)

    for ticker in RU + CU:
        df_RCU[ticker] = pdr.get_data_yahoo(ticker, start_day - timedelta(days=365), end_day)['Adj Close']

    for ticker in BU:
        df_BU[ticker] = pdr.get_data_yahoo(ticker, start_day - timedelta(days=365), end_day)['Adj Close']

    for ticker in CA:
        df_CA[ticker] = pdr.get_data_yahoo(ticker, start_day - timedelta(days=365), end_day)['Adj Close']

    return df_RCU, df_BU, df_CA

# 모멘텀 스코어 뽑는 함수(이건 pdr로 받아와서 바로 해야함)
def get_momentum_ca(x):
    temp_list = [0 for i in range(len(x.index))]
    momentum = pd.Series(temp_list, index=x.index)
    try:
        # print(x.name)
        # print(x)
        # breakpoint()
        # print(df_CA[x.name])
        before1 = df_CA[x.name - timedelta(days=35) : x.name - timedelta(days=30)].iloc[-1][CA]
        before3 = df_CA[x.name - timedelta(days=95) : x.name - timedelta(days=90)].iloc[-1][CA]
        before6 = df_CA[x.name - timedelta(days=185) : x.name - timedelta(days=180)].iloc[-1][CA]
        before12 = df_CA[x.name - timedelta(days=370) : x.name - timedelta(days=365)].iloc[-1][CA]

        momentum = 12 * (x / before1 - 1) + 4 * (x / before3 - 1) + 2 * (x / before6 - 1) + (x / before12 - 1)

    except Exception as e:
        # print("Error : ", str(e))
        pass
    return momentum

# df_RCU = pd.read_csv("./finance/new_data/df_RCU.csv").set_index('Date')
# df_BU = pd.read_csv("./finance/new_data/df_BU.csv").set_index('Date')
# df_CA = pd.read_csv('./finance/new_data/df_CA.csv').set_index('Date')

# # 모멘텀 스코어 뽑는 함수
# def get_momentum(x):
#     temp_list = [0 for i in range(len(x.index))]
#     momentum = pd.Series(temp_list, index=x.index)
#     try:
#         # print(x.name)
#         # print(x)
#         # breakpoint()
#         before1 = df_RCU[x.name - timedelta(days=35) : x.name - timedelta(days=30)].iloc[-1][RU + CU]
#         before3 = df_RCU[x.name - timedelta(days=95) : x.name - timedelta(days=90)].iloc[-1][RU + CU]
#         before6 = df_RCU[x.name - timedelta(days=185) : x.name - timedelta(days=180)].iloc[-1][RU + CU]
#         before12 = df_RCU[x.name - timedelta(days=370) : x.name - timedelta(days=365)].iloc[-1][RU + CU]
#         momentum = 12 * (x / before1 - 1) + 4 * (x / before3 - 1) + 2 * (x / before6 - 1) + (x / before12 - 1)
#     except Exception as e:
#         # print("Error : ", str(e))
#         pass
#     return momentum
#
#
# mom_col_list = [col+'_M' for col in df_RCU[RU+CU].columns]
# df_RCU[mom_col_list] = df_RCU[RU+CU].apply(lambda x: get_momentum(x), axis=1)



# 자산 수익률의 평균을 통해 y값 도출
# 자산군에 대한 등락률 뽑아내기

df_RCU, df_BU, df_CA = get_price_data(RU, CU, BU, CA)

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


mom_col_list = [col+'_M' for col in df_RCU[RU+CU].columns]
df_RCU[mom_col_list] = df_RCU[RU+CU].apply(lambda x: get_momentum(x), axis=1)

profit_col_list = [col+'_P' for col in df_RCU[RU+CU].columns]
df_RCU[profit_col_list] = df_RCU[RU+CU].pct_change()
df_RCU[profit_col_list] = df_RCU[profit_col_list].fillna(0)

# 공격 및 방어 자산 등락률 정규화
scaler = preprocessing.StandardScaler().fit(df_RCU[profit_col_list])
profit_normalize = scaler.transform(df_RCU[profit_col_list])

# 공격+방어자산의 등락률 평균 - 토대 예측시 가장 공격(240번)
y1 = df_RCU[profit_col_list].mean(axis=1)
# 공격+방어자산 등락률에 normalize - 토대 예측시 중간(102번 공격)
y2 = pd.DataFrame(profit_normalize, index=y1.index).mean(axis=1)
# 공격자산의 등락률 평균 - 토대로 예측시 더 보수적으로 운용(89번 공격)
y3 = df_RCU[profit_col_list].iloc[:, :4].mean(axis=1)


# 예전 수집 데이터로 x, y 예측해보기(2012/4/16~2022/4/14)
total_data = pd.read_csv("./finance/total_data.csv")
date = total_data.iloc[:,0]

def process_total(total_data):
    date = total_data.iloc[:,0]
    total_data = total_data.set_index('0')
    # 데이터 정규화
    scaler_td = preprocessing.StandardScaler().fit(total_data)
    total_norm = scaler_td.transform(total_data)
    # x,y값 인덱싱
    X = pd.DataFrame(total_norm, index=date).iloc[:-1]
    y = y3[date.iloc[0]:date.iloc[-1]]
    # 급하게 안 겹치는 데이터 지우기 진행(y기준으로 해서 x값 정리)
    temp_x = X.index.values
    temp_y = y.index.strftime("%Y-%m-%d")

    # x와 y가 날짜 같은 부분의 인덱스 저장(x의 인덱스)
    idx_li = []
    for i in range(len(temp_x)):
        for j in range(len(temp_y)):
            if temp_x[i] == temp_y[j]:
                idx_li.append(i)

    x_ = pd.DataFrame(np.zeros((len(y), np.shape(X)[1])))
    for i, idx in enumerate(idx_li):
        x_.iloc[i] = X.iloc[idx]
    # breakpoint()
    return x_, y, temp_y

# 카나리아 자산군
profit_col = [col+'_P' for col in df_CA[CA].columns]
df_CA[profit_col] = df_CA[CA].pct_change()

mom_col_list = [col+'_M' for col in df_CA[CA].columns]
df_CA[mom_col_list] = df_CA[CA].apply(lambda x: get_momentum_ca(x), axis=1)


# 등락률로 하는 경우와 모멘텀으로 하는 경우 선택
# alarm_asset = df_CA[profit_col].iloc[:,0]
# alarm_asset = df_CA[mom_col_list]


################## 경제지표 데이터
x_econ, y, idx_econ = process_total(total_data)
################## 카나리아 데이터(모멘텀)
# x_alarm = alarm_asset[date.iloc[0]:date.iloc[-1]]


x_cut = x_econ[:-1]
idx_x = idx_econ[:-1]
y_cut = y[1:]

