import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 목표 : 이상적인 sharpe Maximized 포트폴리오 구축 및 결과분석

os.chdir('C://data_minsung/finance/Qraft')

etfs = pd.read_csv('./indexPortfolio/etfs.csv').set_index('Date')
etfs.index = pd.to_datetime(etfs.index)

# EMB 결측치가 3413으로 가장 많다
etfs.isna().sum()

# 12월 31일
# temp_dec = etfs[etfs.index.month == 12]
# yearly_dec = temp_dec[temp_dec.index.day==31]

# # 1월 1일
# temp = etfs[etfs.index.month == 1]
# yearly_jan = temp[temp.index.day == 1]
# yearly_ret = yearly_jan.pct_change()
#
# temp = etfs[etfs.index.year==1994]
#
etfs_ret = etfs.pct_change().dropna()

monthly = etfs.resample('M').last()

year_idx = [monthly.index[12*i] for i in range(len(monthly)//12+1)]
annual = pd.DataFrame(np.zeros((len(monthly)//12+1 , len(monthly.columns))), index=year_idx, columns=monthly.columns)

for i in range(len(monthly)//12):
    annual.iloc[i] = monthly.iloc[(i-1)*12:i*12].mean()


stocks = etfs_ret.columns.values

annual_ret = annual.pct_change().dropna()

etfs_cov = etfs_ret.cov()
annual_cov = annual_ret.cov()

# yearly : 94/2/28~21/2/28 , observation 28개
# montly : 93/2/28~21/5/31, observation 340개

port_ret = []
port_risk = []
port_weights = []
sharpe_ratio = []

stock = etfs_ret.columns.values

# 원래 코드 구조가 1년에 대해서만 구하는 방식, 내껀 27년 해야하므로 for문 추가
i = 0
# for i in range(len(annual_ret)):
    for _ in range(100):
        # 임의의 포트 비중 생성
        weights = np.random.random(len(stock))
        weights /= np.sum(weights)

        returns  = np.dot(weights, annual_ret.iloc[i]) #비중과 수익률 곱해 포트폴리오 수익률 계산
        risk = np.sqrt(np.dot(weights.T, np.dot(annual_cov, weights)))
        SR = returns/risk #Sharpe Ratio

        port_ret.append(returns)
        port_risk.append(risk)
        port_weights.append(weights)
        sharpe_ratio.append(SR)

portfolio = {'Returns': port_ret, 'Risk': port_risk, 'Sharpe': sharpe_ratio}  # portfolio 딕셔너리에 각 종목별로 비중값 추가

for i, s in enumerate(stocks):
    portfolio[s] = [weight[i] for weight in port_weights]  # 개별주식의 weight값 append

portfolio[s]

df = pd.DataFrame(portfolio)
df = df[['Returns', 'Risk', 'Sharpe'] + [s for s in stocks]]
df

# Mean-Varinance portfolio(평균분산포트폴리오) : 최대 샤프지수 ~ 탄젠트
# 첫번째 년도의 max sharpe는 0.643203이 나옴
max_sharpe = df.loc[df['Sharpe'] ==df['Sharpe'].max()]
max_sharpe

# Mininum-Variance portfolio(최소분산포트폴리오) : 최소 Variance(=Risk)
min_risk = df.loc[df['Risk'] == df['Risk'].min()]
min_risk


# minimum-variance portfolio와 mean-variance portfolio 시각화
df.plot.scatter(x='Risk', y='Returns', c='Sharpe', cmap='viridis',
    edgecolors='k', figsize=(11,7), grid=True)

plt.scatter(x=max_sharpe['Risk'], y=max_sharpe['Returns'], c='r',      # 평균분산포트폴리오 *표시
    marker='*', s=300)

plt.scatter(x=min_risk['Risk'], y=min_risk['Returns'], c='r',          # 최소분산포트폴리오 X표시
    marker='X', s=200)

plt.title('Portfolio Optimization')
plt.xlabel('Risk')
plt.ylabel('Expected Returns')
plt.show()

