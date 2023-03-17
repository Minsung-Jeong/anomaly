# import library
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

#For Machine Learning Algorithm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("C://data_minsung/kaggle/online_retail/Online Retail.csv")

df = df.dropna(subset=["CustomerID"])
df = df[(df['Quantity']>0) & (df['UnitPrice']>0)]
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])


"""
corohort analysis on retention & quantity
"""
def get_month(x):
    return dt.datetime(x.year, x.month, 1)

df.columns
# 사용자에 대한 cohort 분석 - 1.최초 구매일, 날짜차이,
df['InvoiceMonth'] = df['InvoiceDate'].apply(get_month)
grouping = df.groupby('CustomerID')['InvoiceMonth']
df['startMonth'] = grouping.transform('min')

def get_date_value(df, col):
    year = df[col].dt.year
    month = df[col].dt.month

    return year, month

start_year, start_month = get_date_value(df, 'startMonth')
invoice_year, invoice_month = get_date_value(df, 'InvoiceMonth')

df['monthDiff'] = (invoice_year-start_year)*12 + invoice_month-start_month+1
df_groupby = df.groupby(['startMonth','monthDiff'])
cohort_data = df_groupby['CustomerID'].apply(pd.Series.nunique)
cohort_data = cohort_data.reset_index()

cohort_table = cohort_data.pivot(index='startMonth', columns='monthDiff', values='CustomerID')

denorminator = cohort_table.iloc[:,0]
# n-day Retention = n-th users / all users
retentaion_table = cohort_table.divide(denorminator, axis=0)
retentaion_table.round(2)*100

# heatmap of retention
plt.figure(figsize=(15,8))
plt.title('Retention cohort')
sns.heatmap(data=retentaion_table, annot=True, fmt = '.0%',vmin = 0.0,vmax = 0.5,cmap="BuPu_r")
plt.show()

# make cohort of quantity
cohort_data_quantity = df_groupby['Quantity'].mean()
cohort_data_quantity = cohort_data_quantity.reset_index()
quantity_table = cohort_data_quantity.pivot(index='startMonth',columns='monthDiff', values='Quantity')
quantity_table.round(1)
quantity_table.index = quantity_table.index.date

# heatmap of quantity
plt.figure(figsize=(15,8))
plt.title('Quantity cohort')
sns.heatmap(data=quantity_table,annot = True,vmin = 0.0, vmax =20, cmap="BuGn_r")
plt.show()

"""
RFM analysis
"""

