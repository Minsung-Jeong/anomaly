import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import solvers
import cvxpy as cvx

# MVO(Mean-Variance Optimization) - Tangency Portfolio
# etfs 결측치는 가장 가까운 값으로 대체(nn에서는 평균값 or 최빈값)
os.chdir('C://data_minsung/finance/Qraft')
etfs = pd.read_csv('./indexPortfolio/etfs.csv')

# Nan값은 가장 가까운 날의 값으로 대체 - 모두 1000인 것을 확인
etfs = etfs.fillna(float(1000))
etfs = etfs.set_index('Date')
etfs_ret = etfs.pct_change(1).dropna()
etfs_cumret = etfs_ret.add(1).cumprod().sub(1)

etfs_cumret.iloc[-1]
np.mean(etfs_ret)

# Minimum variance portfolio (최소 분산 포트폴리오)

returns = etfs_ret.values.T


'''
min 1/2w'COVw
s.t. mu'w = rp and 1'w = 1
'''
def minimum_port_weight(returns):
    mean_ret = np.mean(returns, axis=1)
    cov_ret = np.cov(returns)
    n = mean_ret.shape[0]
    one_array = np.ones(n)
    neg_diag = np.diag(np.diag(-1*np.ones((n,n))))

    # List of target portfolio returns (목표 포트폴리오 수익률 리스트)
    # mu_t = list(np.arange(0, 0.003, 0.0001))
    mu_t = list(np.arange(0, 0.879823, 0.01))

    # Convert to cvxopt matrices (cvxopt matrix로 변환)
    P = opt.matrix(cov_ret)
    q = opt.matrix(0.0, (n, 1))
    # Create constraint matrices (최적화 문제 제약조건 설정)
    # G = opt.matrix(neg_diag)
    # h = opt.matrix(0.0, (n,1))

    A = opt.matrix(np.stack((mean_ret, one_array)))
    weights_list = []
    mean_std_list = []

    # x가 양수라는 조건을 넣으면 convexity 침해 - optimal 값을 구할 수 없으므로 음수는 투자 0로 하고 밸런싱
    for mu in mu_t:
        b = opt.matrix(np.array([mu, 1]))
        """
        minimize(1 / 2) * x'*P*x + q' * x
        subject
        to
        G * x <= h
        A * x = b.
        """

        weights = solvers.qp(P, q,  A=A, b=b)['x']
        weight_array = np.array(list(weights))

        port_ret = np.matrix(weight_array).dot(np.matrix(mean_ret).T)
        port_std = np.sqrt(np.matrix(weight_array).dot(np.matrix(cov_ret)).dot(np.matrix(weight_array).T))

        weights_list.append(list(weights))
        mean_std_list.append([port_ret[0, 0], port_std[0, 0]])

    return weights_list, mean_std_list


weights, mean_stds = minimum_port_weight(etfs_ret.T)
#To dataframe (데이타 프레임으로 변경)
col_name = etfs_ret.columns + '_Weight'
weights_df = pd.DataFrame(weights, columns=col_name)
mean_stds_df = pd.DataFrame(mean_stds,columns=['Port_ret','Port_std'])

# Plot minimum variance frontier (최소 분산 곡선 그래프)

# Minimum varaince portfolios (최소분산 포트폴리오)
opt_returns = mean_stds_df['Port_ret']
opt_stds = mean_stds_df['Port_std']

fig = plt.figure(figsize=(15, 8))
plt.ylabel('mean', fontsize=12)
plt.xlabel('std', fontsize=12)
plt.plot(opt_stds, opt_returns, 'y-o')
plt.title('Minimum variance frontier for risky assets', fontsize=15)


mean_ret = np.mean(returns, axis=1)
cov_ret = np.cov(returns)
n = mean_ret.shape[0]
one_array = np.ones(n)

# 무위험 시장에 따른 Tangency portfolios 생성
rf = 0.0001
mu = np.matrix(mean_ret).T
cov = np.matrix(cov_ret)
cov_inv = np.linalg.inv(cov)

top_no = np.dot(cov_inv, mu)
one_matrix = np.matrix(np.ones(mu.shape[0]))
bottom = np.dot(one_matrix, cov_inv).dot(mu)

tan_weight = top_no / bottom[0,0]
tan_weight.shape

# Sharpe Ratio 계산
tan_port_ret = np.dot(tan_weight.T, mu)
tan_port_std = np.sqrt(np.dot(tan_weight.T, cov).dot(tan_weight))
tan_sharpe = (tan_port_ret[0,0]- rf) / tan_port_std[0,0]

#Generate the third point in the efficient frontier (EFT위의 제3의 점 생성)
tird_eft_std = 0.26 #왜 0.026?
tird_eft_return = tan_sharpe * tird_eft_std+ rf


#Minimum varaince portfolios for risky assets (위험 자산 최소 분산 포트폴리오)
opt_returns = mean_stds_df['Port_ret']
opt_stds = mean_stds_df['Port_std']

# Efficient frontier with a risk free asset (무위험 자산 존재 시 efficient frontier)
rf_tan_ret = np.array([rf, tan_port_ret[0, 0], tird_eft_return])
rf_tan_std = np.array([0, tan_port_std[0, 0], tird_eft_std])

fig = plt.figure(figsize=(15, 8))
plt.ylabel('mean', fontsize=12)
plt.xlabel('std', fontsize=12)
plt.plot(opt_stds, opt_returns, 'y-o')
plt.plot(rf_tan_std, rf_tan_ret, 'y-o', color="red")

plt.title('Mean-Variance Frontier with a risk free asset', fontsize=15)