import pandas as pd
import os
import numpy as np

os.chdir('C://data_minsung/kaggle/store_sales')


# Import
# 당장은 불필요한 데이터
# stores = pd.read_csv("./stores.csv")
#sub = pd.read_csv("./sample_submission.csv")

train = pd.read_csv("./train.csv")
train['date'] = pd.to_datetime(train['date'])
# train.set_index('date', inplace=True)

test = pd.read_csv("./test.csv")
test['date'] = pd.to_datetime(test['date'])
# test.set_index('date', inplace=True)


oil = pd.read_csv("./oil.csv")
oil['date'] = pd.to_datetime(oil['date'])
# oil.set_index('date', inplace=True)


transactions = pd.read_csv("./transactions.csv").sort_values(["store_nbr", "date"])
transactions['date'] = pd.to_datetime(transactions['date'])
# transactions.set_index('date', inplace=True)

holiday = pd.read_csv("./holidays_events.csv")
holiday['date'] = pd.to_datetime(holiday['date'])
# holiday.set_index('date', inplace=True)


np.shape(train)

df_train = train.merge(oil, on='date', how='left')
df_train['dcoilwtico'].isna().sum()
len(df_train)