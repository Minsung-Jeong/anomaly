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
# 개별 이탈 고객은 기하분포, 고객간 이탈 이질성은 베타분포 -> BG 통해서 고객이탈 추정
# nbd : 포아송(개별 고객 구매빈도), 감마(고객간 재구매 차이)를 통해서 ->  생존기간 동안의 기대구매횟수 도출
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(rfm['Frequency'],rfm['Recency'],rfm['T']) # fit = fit dataset to a BG/NBD model

# T동안의 기대구매 횟수 도출
bgf.conditional_expected_number_of_purchases_up_to_time(1,rfm['Frequency'],rfm['Recency'],rfm['T'])\
    .sort_values(ascending=False).head(10)

# 위랑 같은 함수 predict = conditional_expected_number_of_purchases_up_to_time
rfm["buy_1_week_pred"] = bgf.predict(1,rfm['Frequency'],rfm['Recency'],rfm['T'])
rfm["buy_1_month_pred"] = bgf.predict(4,rfm['Frequency'],rfm['Recency'],rfm['T'])

# sort해서 보기
rfm.sort_values("buy_1_month_pred", ascending=False).head()


# rfm["buy_1_month_pred"].sum() # 한 달간 예상되는 총매출
# plot_period_transactions(bgf)
# plt.show()

# gamma-gamma model : expected average profit with gamma-gamma model
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(rfm["Frequency"], rfm["Monetary"])
# the conditional expectation of the average profit per transaction for a group of one or more customers
ggf.conditional_expected_average_profit(rfm["Frequency"], rfm["Monetary"])

rfm["expected_average_profit"] = ggf.conditional_expected_average_profit(rfm["Frequency"], rfm["Monetary"])

# T동안 고객의 기대구매횟수 * 고객당 평균 예상 수익(구매액) * 기간
cltv = ggf.customer_lifetime_value(bgf, rfm['Frequency'], rfm['Recency'], rfm['T'], rfm['Monetary'],
 time=3, # 3 Months
 freq='W', # Frequency of T ,in this case it is 'weekly'
 discount_rate=0.01)
cltv = cltv.reset_index()
cltv_final = rfm.merge(cltv, on="CustomerID", how="left")

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(cltv_final[["clv"]])

cltv_final["scaled_clv"] = scaler.transform(cltv_final[["clv"]])
cltv_final.sort_values(by="scaled_clv", ascending=False).head()
cltv_final["clv_segment"] = pd.qcut(cltv_final["scaled_clv"], 4, labels=["D", "C", "B", "A"])
cltv_final.head()
cltv_final.sort_values(by="scaled_clv", ascending=False).head(10)



cltv_final.sort_values(by="RFM_Score", ascending=False).iloc[:10][['CustomerID', 'clv', 'RFM_Score']]
cltv_final.sort_values(by="clv", ascending=False).iloc[:10][['CustomerID', 'clv', 'RFM_Score']]


