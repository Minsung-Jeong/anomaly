import FinanceDataReader as fdr
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import tensorflow as tf

os.getcwd()
os.chdir('C://data_minsung/finance')


# # 날짜별 데이터
# sp = pd.read_csv('./SP500.csv')
# t10y2y = pd.read_csv('./T10Y2Y_D.csv')
# t10y3m = pd.read_csv('./T10Y3M_D.csv')
# T10Y_D = pd.read_csv('./T10Y_D.csv')
# NASDAQCOM_D = pd.read_csv('./NASDAQCOM_D.csv')
# GOLDVIX_D = pd.read_csv('./GOLDVIX_D.csv')
# DCOILBRENTEU_D = pd.read_csv('./DCOILBRENTEU_D.csv')
# DCOILWTICO_D = pd.read_csv('./DCOILWTICO_D.csv')
#
# # 월별
# producer_price = pd.read_csv('./ProducerPriceAllCom_M.csv')
# consumer_price = pd.read_csv('./ConsumPIAUCSL_M.csv')
# new_order = pd.read_csv('./ACOGNO_M.csv')
# # unemploy = pd.read_csv('./CCSA_W.csv')
# employee = pd.read_csv('./AWHAEMAN_M.csv')
# newhouse = pd.read_csv('./PERMIT_M.csv')
# manufact = pd.read_csv('./AMTMNO_M.csv')
#
#
# def monthToDate(dateDT, monthDT):
#     date_idx = pd.DataFrame({'DATE': dateDT.DATE})
#     temp_value = pd.DataFrame({'value': [None for i in range(len(date_idx))]})
#     df_temp = pd.concat((date_idx, temp_value), axis=1).copy() #copy를 해주지 않으면 포인트 하는 대상 혼선이 있어서 아래 코드에서 값을 못 할당함
#     # 날짜 데이터 4월 16부터 시작, 월 데이터 3월 1일부터 시작
#     df_temp.iloc[0][1] = monthDT.iloc[1][1]
#     # 매월 1일 값 채우는 로직
#     for i in range(len(monthDT.DATE)):
#         date = monthDT.iloc[i, 0]
#         value = monthDT.iloc[i, 1]
#         df_temp.loc[df_temp['DATE'] == date, 'value'] = value
#     # 결측치 전일 데이터로 채우는 로직
#     for j in range(len(date_idx)):
#         if df_temp.iloc[j][1] == None:
#             df_temp.iloc[j][1] = df_temp.iloc[j - 1][1]
#     return df_temp
#
# def csvProcessing(x):
#     input = x.iloc[:,1]
#     Y = []
#     for ob in input:
#         if ob == '.':
#             Y.append(Y[-1])
#         else:
#             Y.append(float(ob))
#     return Y
#
# def scaling_data(input):
#     scaler = preprocessing.StandardScaler()
#     scaler.fit(input)
#     return scaler.transform(input)
#
# pp = monthToDate(dateDT=sp, monthDT=producer_price)
# cp = monthToDate(dateDT=sp, monthDT=consumer_price)
# no = monthToDate(dateDT=sp, monthDT=new_order)
# emp = monthToDate(dateDT=sp, monthDT=employee)
# house = monthToDate(dateDT=sp, monthDT=newhouse)
# manufact = monthToDate(dateDT=sp, monthDT=manufact)
#
# X = sp.DATE
#
# Y = np.array(csvProcessing(sp)).reshape(len(sp.SP500), 1)
# Y2 = np.array(csvProcessing(t10y2y)).reshape(len(sp.SP500), 1)
# Y3 = np.array(csvProcessing(t10y3m)).reshape(len(sp.SP500), 1)
# Y4 = np.array(csvProcessing(NASDAQCOM_D)).reshape(len(sp.SP500), 1)
# Y5 = np.array(csvProcessing(GOLDVIX_D)).reshape(len(sp.SP500), 1)
# Y6 = np.array(csvProcessing(DCOILBRENTEU_D)).reshape(len(sp.SP500), 1)
# Y7 = np.array(csvProcessing(T10Y_D)).reshape(len(sp.SP500), 1)
# Y8 = np.array(csvProcessing(DCOILWTICO_D)).reshape(len(sp.SP500), 1)
# Y9 = np.array(csvProcessing(pp)).reshape(len(sp.SP500), 1)
# Y10 = np.array(csvProcessing(cp)).reshape(len(sp.SP500), 1)
# Y11 = np.array(csvProcessing(no)).reshape(len(sp.SP500), 1)
# Y12 = np.array(csvProcessing(emp)).reshape(len(sp.SP500), 1)
# Y13 = np.array(csvProcessing(house)).reshape(len(sp.SP500), 1)
# Y14 = np.array(csvProcessing(manufact)).reshape(len(sp.SP500), 1)
#
#
# date = (sp.DATE.values).reshape(len(sp.DATE),1)
#
# add_row = np.zeros([1, 15])
# total_data = np.concatenate((date, Y, Y2, Y3, Y4, Y5, Y6, Y7, Y8, Y9, Y10, Y11, Y12, Y13, Y14), axis=1)
#
# # 10배수 만들기 위해 1개 임의 추가
# total_data = np.append(total_data, add_row, axis=0)
# total_data[-1] = total_data[-2]
# total_data[-1,0] = '2022-04-15'
#
# pd.DataFrame(total_data).to_csv('./total_data.csv')


X_load = pd.read_csv('./total_data.csv')
X_load = X_load.values
X = X_load[:, 2:].astype(np.float32)

X_scaled = tf.keras.utils.normalize(X, axis=1)
plt.plot(X_scaled)

"""
Y값 S&P 500에서 normal과 anomal 을 뽑아내는 작업
"""
sp = pd.read_csv('./SP500.csv')

date = sp.DATE.values
date = np.append(date, '2022-04-15')

sp_stock = sp.SP500.values
sp_stock = np.append(sp_stock, sp_stock[-1])

def PointProcessing(x):
    input = x
    Y = []
    for ob in input:
        if ob == '.':
            Y.append(Y[-1])
        else:
            Y.append(float(ob))
    return Y

sp_stock = PointProcessing(sp_stock)


def avg_window(date, price, window_size):
    return price

window_size = 10

# start_date = 윈도우 시작날짜, avg_val = 윈도우만큼 평균낸 수치
# 0503(화) : moving average로 만든 값은 변동성이 극히 낮아져서 보기 힘들지->만 나름의 기준으로 할 수 있
avg_val = []
start_date = []
for i in range(len(sp_stock)-window_size):
    avg_val.append(np.average(sp_stock[i : i + window_size]))
    start_date.append(date[i])

fluctuation = []
for i in range(len(avg_val)-1):
    fluctuation.append((avg_val[i+1] - avg_val[i])/avg_val[i])

"""
anomaly 선정기준 시나리오
1. 코로나로 인한 하락 기간 평균치 : 2020-02-18 ~ 2020-03-20
2. 코로나 하락과 반등 평균치 : 2020-02-18 ~ 2020-06-05
"""
# start_date[2048] = 2020-02-18 ,start_date[2069] = 2020-03-20
anomal_fluct1 = np.abs(fluctuation[2048:2069])
anomal_fluct2 = np.abs(fluctuation[2048:2124])

np.min(anomal_fluct2)

total_mean = np.mean(np.abs(fluctuation[:2048]))
anomal_mean1 = np.mean(anomal_fluct1)
anomal_mean2 = np.mean(anomal_fluct2)

anomal_idx = []
anomaly = []
for i in range(len(fluctuation)):
    if np.abs(fluctuation[i]) > anomal_mean2:
        anomal_idx.append(i)
        anomaly.append(fluctuation[i])


#
anomal_idx_window = []
for idx in anomal_idx:
    for i in range(window_size):
        anomal_idx_window.append(idx+i)

anomal_list = list(set(anomal_idx_window))


# normal =0, anomal =1
new_col = np.zeros(len(date))
for i in anomal_list:
    new_col[i] = 1

date = np.reshape(date, (len(date),1))
sp_stock = np.reshape(sp_stock, (len(sp_stock),1))
new_col = np.reshape(new_col, (len(new_col),1))
last_np = np.concatenate((date, sp_stock, new_col), axis=1)
last_pd = pd.DataFrame(last_np, columns=['DATE', 'SP500', 'STATUS'])
# last_pd.to_csv('./SNP500_anomal2.csv')

len(anomal_list)
# 시각화하는 부분
# anomal_date = []
# for idx in anomal_idx:
#     anomal_date.append(start_date[idx])
#
# len(sp_stock)
# all_stock = pd.DataFrame(sp_stock, date)
# anomal_stock = [sp_stock[x] for x in anomal_idx]
# anomal_stock_pd = pd.DataFrame(anomal_stock, anomal_date)

# plt.plot(all_stock)
# plt.scatter(x=anomal_date, y=anomal_stock, color='limegreen')


