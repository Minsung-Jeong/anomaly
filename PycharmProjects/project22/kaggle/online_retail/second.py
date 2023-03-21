# import library
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions


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
    # return dt.datetime(x.year, x.month, 1)
    return dt.datetime(x.year, x.month, 1)

#

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
df_groupby = df.groupby(['startMonth','monthDiff'])
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

df['TotalSum'] = df['Quantity'] * df['UnitPrice']
print('Min Invoice Date:',df.InvoiceDate.dt.date.min(),'max Invoice Date:',
       df.InvoiceDate.dt.date.max())

# last day + 1
pin_date = df['InvoiceDate'].max() + dt.timedelta(days=1)


rfm = df.groupby(['CustomerID']).agg({'InvoiceDate':lambda x : (pin_date-x.max()).days,
                                      'InvoiceNo':'count',
                                      'TotalSum':'sum'})
rfm.rename(columns={'InvoiceDate':'Recency', 'InvoiceNo':'Frequency', 'TotalSum':'Monetary'}, inplace=True)

T = (pin_date - df.groupby(['CustomerID'])['InvoiceDate'].min()).apply(lambda x:x.days)

# we will set 5 segments of customers
r_level = range(4,0,-1)
f_level = range(1,5)
m_level = range(1,5)

r_quart = pd.qcut(rfm['Recency'], q=4, labels=r_level)
f_quart = pd.qcut(rfm['Frequency'], q=4, labels=f_level)
m_quart = pd.qcut(rfm['Monetary'], q=4, labels=m_level)
rfm = rfm.assign(T = T, R=r_quart, F=f_quart, M=m_quart)

# 고객 등급화하고 이게 매출에 미치는 영향 보기
rfm['RFM_Segment'] = rfm.apply(lambda x:str(int(x['R']))+str(int(x['F']))+str(int(x['M'])),axis=1)
rfm['RFM_Score'] = rfm[['R','F','M']].sum(axis=1)


# ltv 예측 with BG/NBD
"""
churn rate follows Geometric
difference between churn rates follows Beta 

purchase follows poisson 
difference between purchases follows Gamma 
"""
rfm
bgf = BetaGeoFitter(penalizer_coef=0.001) #BG/NBD 모델 중 BG모델 적용 = 고객 이탈에 대한 추정
bgf.fit(rfm['Frequency'],rfm['Recency'],rfm['T'])

# 1 = 1week, top10 customers who are expected to purchase much in a week.
bgf.conditional_expected_number_of_purchases_up_to_time(1,rfm['Frequency'],rfm['Recency'],rfm['T'])\
    .sort_values(ascending=False).head(10)

rfm["buy_1_week_pred"] = bgf.predict(1,rfm['Frequency'],rfm['Recency'],rfm['T'])
rfm["buy_1_month_pred"] = bgf.predict(4,rfm['Frequency'],rfm['Recency'],rfm['T'])

# sort해서 보기
rfm.sort_values("buy_1_month_pred", ascending=False).head()

# 한 달간 예상되는 총매출
rfm["buy_1_month_pred"].sum()

# trn/test 나눠서 해보기