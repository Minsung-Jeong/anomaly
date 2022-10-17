import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
"""
변동성 돌파 통한 단기 스윙 
1. 일봉 기준 range = 전일 고가 - 전일 저가
2. "당일 가격 > 당일시가 + 전일 range" 돌파시점에 시장가 매수, 즉 threshold = 시가+전날 range
3. 다음날 시가 청산 
"""
input = pd.read_csv("C://data_minsung/finance/trade/JPY_KRW_2022.csv").iloc[::-1]
input.columns = ['date', 'close', 'open', 'high', 'low', 'volume', 'volatil']
input["date"] = pd.to_datetime(input["date"])
input.set_index(["date"], inplace=True)
input["range"] = input["high"] - input["low"]

# volatil은 %단위
input["volatil"] = [float(x[:-1]) for x in input["volatil"].values]

# 반만(왜 반만 했지?)
input = input.iloc[100:]

# volatil = pd.DataFrame(input["volatil"].values.reshape(-1,1), index=input.index)

def make_mv(input, w):
    li = []
    for i in range(len(input) - w):
        temp = np.mean(input["open"].values[i:i + w])
        li.append(temp)
    mv = pd.DataFrame(li, index=input.index[w:])
    return mv

# moving average
mv3 =make_mv(input, 3)
mv5 = make_mv(input, 5)
mv10 = make_mv(input,10)

# data visualize
open_df = pd.DataFrame(input["open"].values.reshape(-1,1), index=input.index)
# plt.plot(open_df, color='black')
# plt.plot(mv3, color='red')
# plt.plot(mv5, color='blue')
# plt.plot(mv10, color='green')
# plt.show()


w = 5
total = pd.DataFrame()
total["mv5"] = mv5
total["open"] = open_df[w:]
total["volatil"] = input["volatil"][w:]


mv5
# 접점 찾기 로직(두 곡선 간의 교점)
x = np.arange(0, len(total))


intersections = []
prev_dif = 0
x0, prev_c1, prev_c2 = None, None, None
for x1, c1, c2 in zip(x, mv5.values, open_df[w:].values):
    new_dif = c2 - c1
    if np.abs(new_dif) < 1e-12:
        intersections.append((x1, c1))
    elif new_dif * prev_dif < 0: # 현재 x좌표(x1)에서 y값 차이가 이전 차이와 부호 달라지는 구간
        # 현재 x 좌표(x0)과 이전 x 좌표(x1)사이에 두 곡선의 y값이 같아지는 x 좌표를 찾기 위해 선형 보간(linear interpolation)을 수행합니다.
        # 직선 [(t0, prev_c1), (t1, c1)]과 직선 [(t0, prev_c2), (t1, c2)]의 교차점입니다.
        denom = prev_dif - new_dif
        intersections.append(((-new_dif*x0 + prev_dif*x1)/denom, (c1*prev_c2 - c2*prev_c1)/denom))
    x0, prev_c1, prev_c2, prev_dif = x1, c1, c2, new_dif



fig, ax = plt.subplots()
ax.plot(x, mv5.values, 'b')
ax.plot(x, open_df[w:].values, 'r')
ax.plot(*zip(*intersections), 'go', alpha=0.7, ms=10)


# open이 이평선 아래로 내리고 -5일 때 매수, 이평선 위로 가고 +5이면 매도
# +, - 절대값에 관한 것도 volatility로 해결할 수 있을 것 같음
# mv, open, volatil 순

# 교점 날짜가 전부 float 형태이므로 이를 round 통해 버림(한 시점 빠르게 인식)
# 1. 시가 이평선 아래로 내리고 -5일 때 매수, 구매가<시가, 역전 교점+5 < 시가 이면 매도

inter = np.array(intersections).reshape((-1,2))[:,0]
inter = [math.floor(x) for x in inter]

# 0~1 매도 포인트 / 1~2 매수 포인트 / 2~3 매도 포인트
# 0~1까지의 std를 통해 1 이후 시점의 매수 포인트 잡아보기


# buy = []
# std = np.std(open_df.values[0:inter[0]].reshape(-1))
# for i in range(inter[1], inter[2]):
#     # 지금 가격 < 피봇 - std : buy
#     if total.iloc[i]["open"] < total.iloc[inter[1]]["open"] - std:
#         print(i)
#         buy.append(i)
#         break
#
# sell = []
# std = np.std(open_df.values[inter[1]:inter[2]].reshape(-1))
# for j in range(inter[2], inter[3]):
#     # 지금 가격 > 피봇 + std : sell
#     if total.iloc[j]["open"] > total.iloc[inter[2]]["open"] + std:
#         print(j)
#         sell.append(j)

# 함수로 깔끔하게 정리하기!

# 가격이 이평선 아래에서 매수, 이평선 돌파 후 특정 시점에 매도 전략(100일은 4% 수익, 200은 문제)
buy = []
for i in range(1, len(inter)-1, 2):
    std =  np.std(open_df.values[inter[i-1]:inter[i]].reshape(-1))
    for a in range(inter[i], inter[i+1]):
        # 지금 가격 < 피봇 - std : buy
        if total.iloc[a]["open"] < total.iloc[inter[i]]["open"] - std:
            buy.append(a)
            break

sell = []
for i in range(2, len(inter)-1, 2):
    std =  np.std(open_df.values[inter[i-1]:inter[i]].reshape(-1))
    for a in range(inter[i], inter[i + 1]):
        # 지금 가격 > 피봇 + std : sell
        if total.iloc[a]["open"] > total.iloc[inter[i]]["open"] + std:
            sell.append(a)
            break

action = []
action.extend(buy)
action.extend(sell)

account = []
rev = []

open_val = open_df[w:].values.reshape(-1)

for act in action:
    if act in buy:
        account.append(open_val[act])
    if act in sell:
        while account:
            bought = account.pop()
            rev.append(bought/open_val[act]-1)

