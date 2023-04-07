
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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
rfm = rfm[rfm.Monetary > 0]
T = (pin_date - df.groupby(['CustomerID'])['InvoiceDate'].min()).apply(lambda x:x.days)

# rfm = 1,2,3,4 까지
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
# 4-high, 0-low
rfm['customer_type'] = pd.cut(rfm['RFM_Score'],
                              bins=[0,3,5,7,10,12],
                              labels=[0,1,2,3,4])


# 클러스터링 정독
# https://www.kaggle.com/code/mittalvasu95/cohort-rfm-k-means#III.-k-Means-Clustering

# copying the data into new variable
df_kmeans = rfm.copy()


# # taking only relevant columns
# df_kmeans = df_kmeans.iloc[:,:3]
df_kmeans = df_kmeans.reset_index()


# Removing outliers for Monetary
Q1 = df_kmeans.Monetary.quantile(0.05)
Q3 = df_kmeans.Monetary.quantile(0.95)
IQR = Q3 - Q1
df_kmeans = df_kmeans[(df_kmeans.Monetary >= Q1 - 1.5*IQR) & (df_kmeans.Monetary <= Q3 + 1.5*IQR)]

# Removing outliers for Recency
Q1 = df_kmeans.Recency.quantile(0.05)
Q3 = df_kmeans.Recency.quantile(0.95)
IQR = Q3 - Q1
df_kmeans = df_kmeans[(df_kmeans.Recency >= Q1 - 1.5*IQR) & (df_kmeans.Recency <= Q3 + 1.5*IQR)]

# Removing outliers for Frequency
Q1 = df_kmeans.Frequency.quantile(0.05)
Q3 = df_kmeans.Frequency.quantile(0.95)
IQR = Q3 - Q1
df_kmeans = df_kmeans[(df_kmeans.Frequency >= Q1 - 1.5*IQR) & (df_kmeans.Frequency <= Q3 + 1.5*IQR)]


# data plot after outlier removing
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.scatter(df_kmeans.Recency, df_kmeans.Frequency, color='grey', alpha=0.3)
plt.title('x:Recency, y:Frequency', size=15)
plt.subplot(1,3,2)
plt.scatter(df_kmeans.Monetary, df_kmeans.Frequency, color='grey', alpha=0.3)
plt.title('x:Monetary, y:Frequency', size=15)
plt.subplot(1,3,3)
plt.scatter(df_kmeans.Recency, df_kmeans.Monetary, color='grey', alpha=0.3)
plt.title('x:Recency, y:Monetary', size=15)
plt.show()

# correlation : Monetary, Frequency have high correlation value
import scipy.stats as stats
stats.pearsonr(df_kmeans.Monetary, df_kmeans.Frequency)
stats.pearsonr(df_kmeans.Recency, df_kmeans.Frequency)
stats.pearsonr(df_kmeans.Recency, df_kmeans.Monetary)

# removing customer id as it will not used in making cluster
df_kmeans_id = df_kmeans.iloc[:,0]
df_rfm_clustered = df_kmeans.customer_type
df_kmeans = df_kmeans.iloc[:,1:4]

# scaling the variables and store it in different df
standard_scaler = StandardScaler()
df_kmeans_norm = standard_scaler.fit_transform(df_kmeans)

# converting it into dataframe
df_kmeans_norm = pd.DataFrame(df_kmeans_norm)
df_kmeans_norm.columns = ['recency','frequency','monetary']
df_kmeans_norm.head()

clustered = KMeans(n_clusters = 5)
clustered.fit(df_kmeans_norm)

df_kmeans['kmeans_clusters'] = clustered.labels_
df_kmeans.head()

df_kmeans['rfm_clustered'] = df_rfm_clustered

# rfm 과 k-means clustering 의 상관성은 거의 없음
stats.pearsonr(df_kmeans.rfm_clustered, df_kmeans.kmeans_clusters)



# k-means clustering 도 recency 보다는 freq, monetary 와 더 긴밀한 관계를 맺고 있음
column = ['Recency','Frequency','Monetary']
plt.figure(figsize=(15,4))
for i,j in enumerate(column):
    plt.subplot(1,3,i+1)
    sns.boxplot(y=df_kmeans[j], x=df_kmeans['kmeans_clusters'], palette='spring')
    plt.title('{} wrt clusters'.format(j.upper()), size=13)
    plt.ylabel('')
    plt.xlabel('')
plt.show()

# -------------------------------------------------------
# to-do cltv prediction 진행한 후 r/f/m 과의 관계성 살피기
# Beta-Geometric(BG) fitting
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(rfm['Frequency'],rfm['Recency'],rfm['T']) # fit = fit dataset to a BG/NBD model

# get expected # of purchases up to time T
bgf.conditional_expected_number_of_purchases_up_to_time(1,rfm['Frequency'],rfm['Recency'],rfm['T'])\
    .sort_values(ascending=False).head(10)

# plot_period_transactions(bgf)
# plt.show()

# get parameter 1 = 1 week
# get parameter 4 = 4 weeks = 1 month
# predict = conditional_expected_number_of_purchases_up_to_time
rfm["buy_1_week_pred"] = bgf.predict(1,rfm['Frequency'],rfm['Recency'],rfm['T'])
rfm["buy_1_month_pred"] = bgf.predict(4,rfm['Frequency'],rfm['Recency'],rfm['T'])

rfm.sort_values("buy_1_month_pred", ascending=False).head(10)

# Gamma-Gamma fitting
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(rfm["Frequency"], rfm["Monetary"])

# the conditional expectation of the average profit per transaction for a group of one or more customers
ggf.conditional_expected_average_profit(rfm["Frequency"], rfm["Monetary"])

rfm["expected_average_profit"] = ggf.conditional_expected_average_profit(rfm["Frequency"], rfm["Monetary"])

cltv = ggf.customer_lifetime_value(bgf, rfm['Frequency'], rfm['Recency'], rfm['T'], rfm['Monetary'],
 time=3, # 3 Months
 freq='W', # Frequency of T ,in this case it is 'weekly'
 discount_rate=0.01)

# add cltv columns on rfm => cltv_final
cltv = cltv.reset_index()
cltv_final = rfm.merge(cltv, on="CustomerID", how="left")
cltv_final.sort_values(by="clv", ascending=False).head()

# segment customers with cltv value
cltv_final["clv_segment"] = pd.qcut(cltv_final["clv"], 5, labels=[0,1,2,3,4])


segment = cltv_final[["clv_segment","customer_type"]]
segment["clv_segment"] = segment["clv_segment"].astype(int)
segment["clv_segment"] = segment["customer_type"].astype(int)
# 두 개 넣는 거 그냥 plt, sns 로 해결해서 넣어야 할 듯 
segment.head().plot(kind='bar')