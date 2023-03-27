# import library
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("C://data_minsung/kaggle/e_commerce/data.csv",encoding="ISO-8859-1",
                         dtype={'CustomerID': str,'InvoiceID': str})

df.info()
df.describe()
df.head()



# cohort분석
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['InvoiceMonth'] = df['InvoiceDate'].apply(lambda x: dt.datetime(x.year, x.month,1))
df['startMonth'] = df.groupby('CustomerID')['InvoiceMonth'].transform('min')

def get_date_value(df, col):
    year = df[col].dt.year
    month = df[col].dt.month
    return year, month

start_year, start_month = get_date_value(df, 'startMonth')
invoice_year, invoice_month = get_date_value(df, 'InvoiceMonth')

df['month_diff'] = 12*(invoice_year-start_year) + invoice_month - start_month + 1

# get Customer count
df_groupby = df.groupby(['startMonth','month_diff'])
cohort_data = df_groupby['CustomerID'].apply(pd.Series.nunique)
cohort_data = cohort_data.reset_index()
cohort_table = cohort_data.pivot(index='startMonth', columns='month_diff', values='CustomerID')
denorm = cohort_table.iloc[:,0]
retention_table = cohort_table.divide(denorm, axis=0)

plt.figure(figsize=(15,8))


# 데이터의 추이 및 관계

df.columns
quantity = df.groupby('CustomerID')['Quantity'].mean()
country = df.groupby('CustomerID')['Country'].min()

