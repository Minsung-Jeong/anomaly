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
plt.plot(open_plot, color='black')

"""
완성코드
# 로직 검증 필요
# 21년에 대해서도 가능한지 확인 -> 12%의 수익률
# 기준 가격과 diff에 대해서 window 계속 움직일 수 있게 해야함
"""
open_val = input["open"].values
open_idx = input.index

high_val = input["high"].values
low_val =  input["low"].values

middle_val = np.average((high_val, low_val), axis=0)




split = 5
# 환율 우대 80%~95%
comisison = 20*(1-0.95) / 100


trade = 0
result = []
account = []
account_rec = []
buy_idx = []
buy_pr = []
sell_idx = []
sell_pr = []
account_with_date = pd.DataFrame(np.zeros((len(input), split)), index=input.index)

for i in range(30, len(open_val)):
    temp = input["open"].iloc[i-30:i]
    max_idx, min_idx, mdd = get_mdd(temp)
    pric_standard = np.mean(temp)
    # diff_standard = abs(mdd) / split
    diff_standard = abs(mdd) / split * open_val[i-1]
    # diff_standard = 0.05

    # 최초매수 : 계좌에 외화가 없을 때
    if open_val[i] < pric_standard and len(account) == 0 and open_val[i] < 9.9:
    # if open_val[i] < pric_standard and len(account) == 0:
        account.append(open_val[i])
        buy_idx.append(open_idx[i])
        buy_pr.append(open_val[i])
        trade += 1
        print("start buy")
    # 매수로직 : 가격 더 떨어지면 구매, 5개 미만일 때 구매
    if 0< len(account) < split and (low_val[i] < account[-1]-diff_standard < high_val[i]) and open_val[i] < 9.9:
    # if 0< len(account) < split and account[-1]-diff_standard > open_val[i]:
        buy_pr.append(account[-1] - diff_standard)
        account.append(account[-1]-diff_standard)
        buy_idx.append(open_idx[i])

        trade += 1
    # 매도로직 : low < 구매+차이+수수료 < high
    if account and low_val[i] < account[-1]+diff_standard+comisison < high_val[i]:
        sell_pr.append(account[-1] + diff_standard + comisison)
        rev = (account[-1] + diff_standard)/account.pop()-1
        result.append(rev)
        sell_idx.append(open_idx[i])

        trade += 1

    account_with_date.iloc[i, :len(account)] = account

a = pd.DataFrame(buy_pr, index=buy_idx)
b = pd.DataFrame(sell_pr, index=sell_idx)

plt.plot(open_plot, color='black')
plt.scatter(a.index, a.values, color='red')
plt.scatter(b.index, b.values, color='blue')
plt.show()

# 계좌잔액 손익률
recent = open_val[-1]
account_rev = sum([(recent/x-1) for x in account])/ split

print(np.sum(result)/split+account_rev, trade)


# table = pd.concat((a,b, account_with_date), axis=1)
# table.to_csv("C://data_minsung/finance/trade/result/result2022_detail.csv")



