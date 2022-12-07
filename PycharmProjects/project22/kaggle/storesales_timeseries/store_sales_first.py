import pandas as pd
import os
import numpy as np
from sklearn.impute import KNNImputer

os.chdir('C://data_minsung/kaggle/store_sales')

# Import
# 당장은 불필요한 데이터
#sub = pd.read_csv("./sample_submission.csv")

train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")
oil = pd.read_csv("./oil.csv")
transactions = pd.read_csv("./transactions.csv").sort_values(["store_nbr", "date"])
holiday = pd.read_csv("./holidays_events.csv")
stores = pd.read_csv("./stores.csv") # store는 시간에 종속적이지 않은 데이터인데 유의미할까? - store_nbr에 종속적


df_train = train.merge(oil, on='date', how='left')
df_train = df_train.merge(transactions, on=['date', 'store_nbr'], how='left') # transaction 좀 더 고민해보기
df_train = df_train.merge(holiday, on=['date'], how='left')
df_train = df_train.merge(stores, on=['store_nbr'], how='left')

# 날짜 데이터 변경 후  set index
df_train['date'] = pd.to_datetime(df_train['date'])
df_train.set_index('date', inplace=True)

# family 컬럼 수치화
dic_family = {}
for i, x in enumerate(list(set(df_train['family'])), start=0):
    dic_family[x] = i

dic_type_x = {}
for i, x in enumerate(list(set(df_train['type_x'])), start=0):
    dic_type_x[x] = i



df_train.columns
for i in range(len(df_train)):
    df_train['family'][i] = dic_family[df_train['family'][i]]

    df_train['type_x'][i] = dic_type_x[df_train['type_x'][i]]
