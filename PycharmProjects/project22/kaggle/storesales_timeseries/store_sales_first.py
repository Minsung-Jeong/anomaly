import pandas as pd
import os
import numpy as np
from sklearn.impute import KNNImputer
import statsmodels.api as sm
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import tensorflow as tf


# 정규화 -> (이상치제거) -> 데이터보간
os.chdir('C://data_minsung/kaggle/store_sales')

# # Import
# # 당장은 불필요한 데이터
# #sub = pd.read_csv("./sample_submission.csv")
#
# train = pd.read_csv("./train.csv")
# test = pd.read_csv("./test.csv")
# oil = pd.read_csv("./oil.csv")
# transactions = pd.read_csv("./transactions.csv").sort_values(["store_nbr", "date"])
# holiday = pd.read_csv("./holidays_events.csv")
# holiday.drop('description',axis=1, inplace=True)
# stores = pd.read_csv("./stores.csv") # store는 시간에 종속적이지 않은 데이터인데 유의미할까? - store_nbr에 종속적
#
#
# # 이것도 나중에 - 일단 있는 데이터로 분석
# df_train = train.merge(oil, on='date', how='left')
# df_train = df_train.merge(transactions, on=['date', 'store_nbr'], how='left') # transaction 좀 더 고민해보기
# df_train = df_train.merge(holiday, on=['date'], how='left')
# df_train = df_train.merge(stores, on=['store_nbr'], how='left')
#
#
#
# def get_dict(data):
#     dic_x = {}
#     for i, x in enumerate((data), start=0):
#         dic_x[x] = i
#     return dic_x
#
#
# # 비정형 데이터 정형화
# dic_family = get_dict(list(set(df_train['family'])))
# dic_type_x = get_dict(list(set(df_train['type_x'])))
# dic_local = get_dict(list(set(df_train['locale'])))
# dic_local_name = get_dict(list(set(df_train['locale_name'])))
# dic_transferred = get_dict(list(set(df_train['transferred'])))
# dic_city = get_dict(list(set(df_train['city'])))
# dic_state = get_dict(list(set(df_train['state'])))
# dic_type_y = get_dict(list(set(df_train['type_y'])))
#
#
# df_train.drop('id',axis=1, inplace=True)
# df_train.date = pd.to_datetime(df_train.date)
# df_train.set_index('date', inplace=True)
#
# np_train = df_train.values.copy()
#
#
# for i in range(len(np_train)):
#     # family
#     np_train[i][1] = dic_family[np_train[i][1]]
#     np_train[i][6] = dic_type_x[np_train[i][6]]
#     np_train[i][7] = dic_local[np_train[i][7]]
#     np_train[i][8] = dic_local_name[np_train[i][8]]
#     np_train[i][9] = dic_transferred[np_train[i][9]]
#     np_train[i][10] = dic_city[np_train[i][10]]
#     np_train[i][11] = dic_state[np_train[i][11]]
#     np_train[i][12] = dic_type_y[np_train[i][12]]
#
#     # np_train['family'][i] = dic_family[df_train['family'][i]]
#     # np_train['type_x'][i] = dic_type_x[df_train['type_x'][i]]
#     # np_train['locale'][i] = dic_local[df_train['locale'][i]]
#     # np_train['locale_name'][i] = dic_local_name[df_train['locale_name'][i]]
#     # np_train['transferred'][i] = dic_transferred[df_train['transferred'][i]]
#     # np_train['city'][i] = dic_city[df_train['city'][i]]
#     # np_train['state'][i] = dic_state[df_train['state'][i]]
#     # np_train['type_y'][i] = dic_type_y[df_train['type_y'][i]]
#
# df_train_aft = pd.DataFrame(np_train, index=df_train.index, columns=df_train.columns)
# df_train_aft.to_csv("./train_processed.csv")

# sales가 label

df_train_aft = pd.read_csv("./train_processed.csv")
df_train_aft['date'] = pd.to_datetime(df_train_aft['date'])
df_train_aft.set_index('date', inplace=True)

df_train_aft.head()
df_train_aft.describe()
df_train_aft.columns

# 데이터 수가 많기 때문에 결측치는 drop(3054348->1933305)
df_train_aft = df_train_aft.dropna()
# sales와 가장 상관성 높은 변수 : 'onpromotion', 'transactions'이 예측에 가장 중요할 것으로 예상
sns.heatmap(df_train_aft.corr(method='pearson'),annot=True)
plt.shop()

x = df_train_aft.drop('sales', axis=1)
y = df_train_aft['sales']

def normalize(data):
    mean = data.mean()
    std = data.std()
    return (data-mean)/std

def remove_out(data, remove_col):
    inp = data.copy()
    idx = []
    for k in remove_col:
        col_data = inp[k]
        Q1 = inp[k].quantile(0.25)
        Q3 = inp[k].quantile(0.75)
        IQR = Q3 - Q1
        IQR = IQR*2 #2배수에 대한 논리 필요
        low = Q1 - IQR
        high = Q3 + IQR
        outlier_idx = col_data[((col_data < low) | (col_data > high))].index
        inp.drop(outlier_idx, axis=0, inplace=True)
        idx.extend(outlier_idx)
    return inp, idx

x_norm = normalize(x)


"""
여러 가지 모델 이용해보기
"""


# import Random Forest classifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import time

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
rfc = RandomForestRegressor(random_state=0)
# 시작시간
start = time.time()
rfc.fit(X_train, y_train)
# 종료시간
end = time.time()
print(f"{end - start:.5f} sec")
y_pred = rfc.predict(X_test)
mean_absolute_error(y_test, y_pred)


y_pred_plt = pd.DataFrame(y_pred, index=range(len(y_pred)))
y_test_plt = pd.DataFrame(y_test.values, index=range(len(y_test)))
plt.plot(y_pred_plt[:100], color='b')
plt.plot(y_test_plt[:100],  color='r')
plt.show()