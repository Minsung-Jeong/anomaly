import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
"""
전략2 : 
최근 1달 동안의 mdd를 바탕
mdd를 split하고 해당 split값 마다 매수 매도 반복 
"""
def get_mdd(x):
    """
    MDD(Maximum Draw-Down)
    :return: (peak_upper, peak_lower, mdd rate)
    """
    arr_v = np.array(x)
    peak_lower = np.argmax(np.maximum.accumulate(arr_v) - arr_v)
    peak_upper = np.argmax(arr_v[:peak_lower])
    return peak_upper, peak_lower, (arr_v[peak_lower] - arr_v[peak_upper]) / arr_v[peak_upper]

input = pd.read_csv("C://data_minsung/finance/trade/JPY_KRW_2022.csv").iloc[::-1]
input.columns = ['date', 'close', 'open', 'high', 'low', 'volume', 'volatil']
input["date"] = pd.to_datetime(input["date"])
input.set_index(["date"], inplace=True)

# plot open data
open_plot = pd.DataFrame(input["open"].values.reshape(-1,1), index=input.index)
plt.plot(open_plot)
temp_plot = open_plot[-60:-30]
plt.plot(temp_plot)

# 기준 가격과 diff에 대해서 window 계속 움직일 수 있게 해야함
temp = input["open"].iloc[-60:-30]


max_idx, min_idx, mdd = get_mdd(temp)

pric_standard = np.mean(temp)

split = 5
diff_standard = abs(mdd)/split

account = []
trade = 0
rev_li = []
# price standard 보다 낮으면 매수 시작 -> diff_standard 기준으로 내려갈 떄마다 매수
# n번 나눠서 진행(ex:1000만원 200만원씩 5번)
for i in range(len(temp)):
    # 최초매수 : 계좌에 외화가 없을 때
    if temp.values[i] < pric_standard and len(account) == 0:
        account.append(temp.values[i])
        trade += 1
        print("start buy")
    # 매수로직 : 가격 더 떨어지면 구매, 5개 미만일 때 구매
    if 0< len(account) < split and account[-1]-diff_standard > temp.values[i]:
        account.append(temp.values[i])
        trade += 1
    # 매도로직 : 가장 최근 매수가격 보다 오르면 매도
    if 0 < len(account) and temp.values[i] > account[-1]+diff_standard:
        rev = temp.values[i]/account.pop()-1
        rev_li.append(rev)
        trade += 1

