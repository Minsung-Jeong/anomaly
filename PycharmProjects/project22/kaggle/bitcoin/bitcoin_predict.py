import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C://data_minsung/kaggle/bitcoin/bitcoin.csv")


df.Timestamp = pd.to_datetime(df.Timestamp, unit='s')

df.set_index('Timestamp', inplace=True)


df = df.resample('D').mean()
df_month =df.resample('M').mean()
df_year = df.resample('Y').mean()
df_year.iloc[:3]


df.columns
# plot data

fig = plt.figure(figsize=[16,8])

plt.subplot(221)
plt.title('Bitcoin price, USD', fontsize=20)
plt.plot(df["Weighted_Price"], label='Daily')
plt.legend()

plt.subplot(222)
plt.title('Bitcoin price, USD', fontsize=20)
plt.plot(df_month["Weighted_Price"],'-', label='Monthly')
