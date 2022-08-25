import pandas_datareader as pdr
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import math
import quantstats as qs


start_day = datetime(2020,1,1) # 시작일
end_day = datetime(2022,8,23) # 종료일

asset_df = pdr.get_data_yahoo('IAU', start_day - timedelta(days=365), end_day)
asset_price = asset_df['Adj Close']

window = 20

asset_mv = [ asset_price.values[i:i+window].mean() for i in range(len(asset_price)-window)]
asset_mv = pd.DataFrame(asset_mv, index=asset_df.index[window:])

plt.plot(asset_price)
plt.plot(asset_mv)

asset_month = asset_price.resample('M').last()
asset_mon_val = asset_month.values

mom_score = []
score_avg = 0
mom_len = 11
for i in range(mom_len):
    var = (asset_mon_val[-1] - asset_mon_val[-1-i]) / asset_mon_val[-1-i]
    if var > 0:
        score_avg += 1 / mom_len
    mom_score.append(var)

asset_mon_val
plt.plot(asset_month[-12:])


# 첫날 200%, 둘째날 -80%
1.6*0.76
1.2*0.92

126920 - 124610