# import library
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

#For Machine Learning Algorithm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# df = pd.read_excel("C://data_minsung/kaggle/online_retail/Online Retail.xlsx")
df = pd.read_csv("C://data_minsung/kaggle/online_retail/Online Retail.csv")

df = df.dropna(subset=["CustomerID"])
df = df[(df['Quantity']>0) & (df['UnitPrice']>0)]



# cohort analysis
def get_month(x):
    return dt.datetime(x.year ,x.month,1)

# invoiceMonth 그냥 구매일, cohorMonth 최초 구매일
df['InvoiceMonth'] = df['InvoiceDate'].apply(get_month)
grouping = df.groupby('CustomerID')['InvoiceMonth']
# transform은 row의 원형유지 - sql의 window function 느낌
df['CohortMonth'] = grouping.transform('min')


def get_month_int(dframe, column):
    year = dframe[column].dt.year
    month = dframe[column].dt.month
    day = dframe[column].dt.day
    return year, month , day

invoice_year, invoice_month,_ = get_month_int(df,'InvoiceMonth')
cohort_year, cohort_month,_ = get_month_int(df,'CohortMonth')

year_diff = invoice_year - cohort_year
month_diff = invoice_month - cohort_month

# cohortIndex = (구매일 - 첫구매일) + 1
df['CohortIndex'] = year_diff * 12 + month_diff + 1

#Count monthly active customers from each cohort(여기서 col 'CustomerID'의 값은 고객수)
grouping = df.groupby(['CohortMonth', 'CohortIndex'])
cohort_data = grouping['CustomerID'].apply(pd.Series.nunique)

# Return number of unique elements in the object.
cohort_data = cohort_data.reset_index()

# pivot은 주어진 index, columns으로 df를 reshape 해주는 것
cohort_counts = cohort_data.pivot(index='CohortMonth',columns='CohortIndex',values='CustomerID')

# Retention table
cohort_size = cohort_counts.iloc[:,0]
retention = cohort_counts.divide(cohort_size,axis=0) #axis=0 to ensure the divide along the row axis
retention.round(3) * 100 #to show the number as percentage

#Build the heatmap
plt.figure(figsize=(15, 8))
plt.title('Retention rates')
sns.heatmap(data=retention,annot = True,fmt = '.0%',vmin = 0.0,vmax = 0.5,cmap="BuPu_r")
plt.show()

#Average quantity for each cohort
grouping = df.groupby(['CohortMonth', 'CohortIndex'])
cohort_data = grouping['Quantity'].mean()
cohort_data = cohort_data.reset_index()
average_quantity = cohort_data.pivot(index='CohortMonth',columns='CohortIndex',values='Quantity')
average_quantity.round(1)
average_quantity.index = average_quantity.index.date

#Build the heatmap
plt.figure(figsize=(15, 8))
plt.title('Average quantity for each cohort')
sns.heatmap(data=average_quantity,annot = True,vmin = 0.0, vmax =20, cmap="BuGn_r")
plt.show()

"""
# Recency, Frequency, Monetary Value Calculation(RFM)
# recency : 고객의 마지막 구매 , frequency : 일정기간 구매횟수, monetary : 고객이 지불한 금액

# Process of calculating percentiles:
    # Sort customers based on that metric
    # Break customers into a pre-defined number of groups of equal size
    # Assign a label to each group
"""

# New Total Sum Column
df['TotalSum'] = df['UnitPrice'] * df['Quantity']

# Data preparation steps
print('Min Invoice Date:',df.InvoiceDate.dt.date.min(),'max Invoice Date:',
       df.InvoiceDate.dt.date.max())

df.head(3)

snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
snapshot_date

"""
#The last day of purchase in total is 09 DEC, 2011. To calculate the day periods, 
#let's set one day after the last one,or 
#10 DEC as a snapshot_date. We will found the diff days with snapshot_date.
"""
# Calculate RFM metrics
rfm = df.groupby(['CustomerID']).agg({'InvoiceDate': lambda x : (snapshot_date - x.max()).days,
                                      'InvoiceNo':'count','TotalSum': 'sum'})
#Function Lambdea: it gives the number of days between hypothetical today and the last transaction

#Rename columns
rfm.rename(columns={'InvoiceDate':'Recency','InvoiceNo':'Frequency','TotalSum':'MonetaryValue'}
           ,inplace= True)

#Final RFM values
rfm.head()


#Building RFM segments
r_labels =range(4,0,-1)
f_labels=range(1,5)
m_labels=range(1,5)
r_quartiles = pd.qcut(rfm['Recency'], q=4, labels = r_labels)
f_quartiles = pd.qcut(rfm['Frequency'],q=4, labels = f_labels)
m_quartiles = pd.qcut(rfm['MonetaryValue'],q=4,labels = m_labels)
rfm = rfm.assign(R=r_quartiles,F=f_quartiles,M=m_quartiles)

# Build RFM Segment and RFM Score
def add_rfm(x) : return str(x['R']) + str(x['F']) + str(x['M'])
rfm['RFM_Segment'] = rfm.apply(add_rfm,axis=1 )
rfm['RFM_Score'] = rfm[['R','F','M']].sum(axis=1)

rfm.head()