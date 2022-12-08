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
holiday.drop('description',axis=1, inplace=True)
stores = pd.read_csv("./stores.csv") # store는 시간에 종속적이지 않은 데이터인데 유의미할까? - store_nbr에 종속적


# 이것도 나중에 - 일단 있는 데이터로 분석
# df_train = train.merge(oil, on='date', how='left')
# df_train = df_train.merge(transactions, on=['date', 'store_nbr'], how='left') # transaction 좀 더 고민해보기
# df_train = df_train.merge(holiday, on=['date'], how='left')
# df_train = df_train.merge(stores, on=['store_nbr'], how='left')



def get_dict(data):
    dic_x = {}
    for i, x in enumerate((data), start=0):
        dic_x[x] = i
    return dic_x


# 비정형 데이터 정형화
dic_family = get_dict(list(set(df_train['family'])))
dic_type_x = get_dict(list(set(df_train['type_x'])))
dic_local = get_dict(list(set(df_train['locale'])))
dic_local_name = get_dict(list(set(df_train['locale_name'])))
dic_transferred = get_dict(list(set(df_train['transferred'])))
dic_city = get_dict(list(set(df_train['city'])))
dic_state = get_dict(list(set(df_train['state'])))
dic_type_y = get_dict(list(set(df_train['type_y'])))

#
# # # 집 컴퓨터로 돌리기 - 요거는 일단 생략하고 데이터 돌려보자
# for i in range(len(df_train)):
#     df_train['family'][i] = dic_family[df_train['family'][i]]
#     df_train['type_x'][i] = dic_type_x[df_train['type_x'][i]]
#     df_train['locale'][i] = dic_local[df_train['locale'][i]]
#     df_train['locale_name'][i] = dic_local_name[df_train['locale_name'][i]]
#     df_train['transferred'][i] = dic_transferred[df_train['transferred'][i]]
#     df_train['city'][i] = dic_city[df_train['city'][i]]
#     df_train['state'][i] = dic_state[df_train['state'][i]]
#     df_train['type_y'][i] = dic_type_y[df_train['type_y'][i]]

df_train.columns

