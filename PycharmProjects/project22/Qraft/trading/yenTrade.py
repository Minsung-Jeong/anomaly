import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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

# 반만
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
plt.plot(open_df, color='black')
plt.plot(mv3, color='red')
plt.plot(mv5, color='blue')
plt.plot(mv10, color='green')
plt.show()


# open이 이평선 아래로 내리고 -5일 때 매수, 이평선 위로 가고 +5이면 매도
# +, - 절대값에 관한 것도 volatility로 해결할 수 있을 것 같음
# mv, open, volatil 순
w = 5
total = pd.DataFrame()
total["mv5"] = mv5
total["open"] = open_df[w:]
total["volatil"] = input["volatil"][w:]


# 접점 찾기 로직
x = np.arange(0, len(total))


intersections = []
prev_dif = 0
x0, prev_c1, prev_c2 = None, None, None
for x1, c1, c2 in zip(x, mv5.values, )


# up = []
# down = []
# idx = total.index
# for i in range(len(total)):
#
#     if total.iloc[i]["mv5"] > total.iloc[i]["open"]:
#         down.append(idx[i])
#     else:
#         up.append(idx[i])


# 두 곡선 간의 교점 찾는 방법
x = np.linspace(0, 10, 40)
curve1 = -np.cos(x+10)
curve2 = -np.cos(x)
fig, ax = plt.subplots()
# 두 곡선을 각각 파란색, 빨간색으로 그립니다.
ax.plot(x, curve1,'b')
ax.plot(x, curve2,'r')

intersections = []
prev_dif = 0
x0, prev_c1, prev_c2 = None, None, None
for x1, c1, c2 in zip(x, curve1, curve2):  # 현재 x좌표에 해당되는 두 곡선의 y 좌표가 각각 c1, c2입니다.
    new_dif = c2 - c1 # 현재 x 좌표에서 두 곡선의 y좌표의 차이입니다.
    if np.abs(new_dif) < 1e-12: # 현재 x 좌표에서 두 곡선의 y좌표 차이가 0인 경우입니다. 즉 두 곡선이 교차하는 지점입니다. 차이가 0인 경우로 체크하지 않고 이렇게 하는군요.
        intersections.append((x1, c1))
    elif new_dif * prev_dif < 0:  # 현재 x 좌표(x1)에서 두 곡선의 y좌표 차이가 이전 x좌표(x0)에서 계산한 두 곡선의 y좌표 차이의 부호와 다른 순간입니다. 즉 부호가 바뀌는 순간입니다.
                                  # 즉 현재 x 좌표(x0)과 이전 x 좌표(x1)사이에 두 곡선의 교점이 있는 것을 알 수 있습니다.
        # 현재 x 좌표(x0)과 이전 x 좌표(x1)사이에 두 곡선의 y값이 같아지는 x 좌표를 찾기 위해 선형 보간(linear interpolation)을 수행합니다.
        # 직선 [(t0, prev_c1), (t1, c1)]과 직선 [(t0, prev_c2), (t1, c2)]의 교차점입니다.
        denom = prev_dif - new_dif
        intersections.append(((-new_dif*x0  + prev_dif*x1) / denom, (c1*prev_c2 - c2*prev_c1) / denom))
    x0, prev_c1, prev_c2, prev_dif = x1, c1, c2, new_dif

print(intersections) # 두 곡선의 교점 리스트를 출력합니다.

ax.plot(*zip(*intersections), 'go', alpha=0.7, ms=10) # 두 곡선의 교점을 그래프에 녹색점으로 출력합니다.
