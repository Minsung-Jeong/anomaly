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

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['InvoiceMonth'] = df['InvoiceDate'].apply(lambda x: dt.datetime(x.year, x.month,1))

# 데이터의 추이 및 관계(봄,여름, 12월(크리스마스, 신년) 시즌에 높은 sales-> 영향 높은 변수는 unit price)
# salesTotal : unit Price * Quantity
# sales is low from september to November
# may and june is the peak month of sales(total sales)

# ironically August is the peak month of quantity
# unit price must be low in september
# I can't jump to conclusion because there is only one year of observations.
df['SalesTotal'] = df['Quantity'] * df['UnitPrice']
month_sales = df.groupby(['InvoiceMonth'])['Quantity','UnitPrice','SalesTotal'].mean()
# month_sales = temp.set_index('InvoiceMonth')
month_sales.plot( subplots=True, rot=0, figsize=(9, 7), layout=(1, 3))
plt.tight_layout()
plt.show()

# cohort분석 - 10년 12월, 11년 1월의 고객이 높은 충성도를 보임
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

# handle startMonth to get rid of hours on sns.heatmap
retention_table = retention_table.reset_index()
retention_table.startMonth = pd.to_datetime(retention_table.startMonth)
retention_table.startMonth = retention_table.startMonth.dt.date
retention_table = retention_table.set_index('startMonth')
#

plt.figure(figsize=(15,8))
plt.title('User Cohort')
ax = sns.heatmap(data=retention_table, annot=True, fmt = '.0%',vmin = 0.1,vmax = .8,cmap="BuPu_r")
plt.show()


# make cohort of quantity
cohort_quantity_data = df_groupby['Quantity'].mean()
cohort_data_quantity = cohort_quantity_data.reset_index()
quantity_table = cohort_data_quantity.pivot(index='startMonth',columns='month_diff', values='Quantity')

plt.figure(figsize=(15,8))
plt.title('Quantity cohort')
sns.heatmap(data=quantity_table,annot = True,vmin = 0.0, vmax =20, cmap="BuGn_r")
plt.show()


"""
RFM analysis & k-mean clustering
customer segments with RFM or K-means clustering
"""

pin_date = df['InvoiceDate'].max() + dt.timedelta(days=1)

rfm = df.groupby(['CustomerID']).agg({
    'InvoiceDate':lambda x : (pin_date-x.max()).days,
    'InvoiceNo' : 'count',
    'SalesTotal' : 'sum'
})
rfm = rfm.rename(columns={'InvoiceDate':'Recency','InvoiceNo':'Frequency','SalesTotal':'Monetary'})

T = (pin_date - df.groupby(['CustomerID'])['InvoiceDate'].min()).apply(lambda x:x.days)

# 1,2,3,4 까지
r_level = range(4,0,-1)
f_level = range(1,5)
m_level = range(1,5)


r_quart = pd.qcut(rfm['Recency'], q=4, labels=r_level)
f_quart = pd.qcut(rfm['Frequency'], q=4, labels=f_level)
m_quart = pd.qcut(rfm['Monetary'], q=4, labels=m_level)
rfm = rfm.assign(T = T, R=r_quart, F=f_quart, M=m_quart)

# 고객 등급화
rfm['RFM_Segment'] = rfm.apply(lambda x:str(int(x['R']))+str(int(x['F']))+str(int(x['M'])),axis=1)
rfm['RFM_Score'] = rfm[['R','F','M']].sum(axis=1)
# 3<= x <5, 5<= x < 7, 7<=x<10, 10<=x<12, x=12
rfm.RFM_Score.describe(percentiles=[0.25,0.5,0.75,0.9,0.95,0.99])

rfm['customer_type'] = pd.cut(rfm['RFM_Score'],
                              bins=[0,3,5,7,10,12],
                              labels=['Iron','Bronze','Silver','Gold','Platinum'])


# 클러스터링 정독
# https://www.kaggle.com/code/mittalvasu95/cohort-rfm-k-means#III.-k-Means-Clustering