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
# Nan_ticker = ['TLT','EMB','Cash','VWO','DBC']
# fill_val = []
# for ticker in Nan_ticker:
#     for x in etfs[ticker]:
#         if not math.isnan(x):
#             fill_val.append(x)
#             break

etfs = etfs.fillna(float(1000))

etfs = etfs.set_index('Date')
etfs_ret = etfs.pct_change(1).T

etfs_cumret = etfs_ret.add(1).cumprod().sub(1)*100



# def rand_weights(n):
#     ''' Produces n random weights that sum to 1 '''
#     k = np.random.rand(n)
#     return k / sum(k)
#
# # Generate a random portfolio (무작위 포트폴리오 생성)
# def random_portfolio(returns):
#     """
#         Returns the mean and standard deviation of returns for a random portfolio
#     """
#     p = np.asmatrix(np.mean(returns, axis=1))
#     w = np.asmatrix(rand_weights(returns.shape[0]))
#     C = np.asmatrix(np.cov(returns))
#
#     mu = w * p.T
#     sigma = np.sqrt(w * C * w.T)
#
#     # This recursion reduces outliers to keep plots pretty
#     if sigma > 2:
#         return random_portfolio(returns)
#     return mu, sigma, w


# Minimum variance portfolio (최소 분산 포트폴리오)
'''
min 1/2w'COVw
s.t. mu'w = rp and 1'w = 1
'''
returns = etfs_ret.values[:,1:]

x = np.reshape([1,2,-3,4,5,6,7,8,-9], (9,1))
neg_diag@x <= 0

def minimum_port_weight(returns):

    mean_ret = np.mean(returns, axis=1)
    cov_ret = np.cov(returns)
    n = mean_ret.shape[0]
    one_array = np.ones(n)
    neg_diag = np.diag(np.diag(-1*np.ones((n,n))))

    # List of target portfolio returns (목표 포트폴리오 수익률 리스트)
    mu_t = list(np.arange(0, 0.003, 0.0001))

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

        # Objective Function
        # min Var(r_p) = w'Covw
        # s.t. return_p = return_i, x >= 0

        # #####################임시
        # H = cov_ret
        # X = cvx.Variable((9, 1))
        #
        # zeros = np.zeros((n, 1))
        # B = np.stack((mean_ret, one_array))
        # R = np.array([mu,1]).reshape(2,1)
        #
        #
        # obj = cvx.Minimize(cvx.quad_form(X, H))
        # constraints = [zeros <= X, B*X==R] #
        # prob = cvx.Problem(obj, constraints)
        # prob.solve()
        # X.value
        # # ############################

        weights = solvers.qp(P, q,  A=A, b=b)['x']
        weight_array = np.array(list(weights))

        port_ret = np.matrix(weight_array).dot(np.matrix(mean_ret).T)
        port_std = np.sqrt(np.matrix(weight_array).dot(np.matrix(cov_ret)).dot(np.matrix(weight_array).T))

        weights_list.append(list(weights))
        mean_std_list.append([port_ret[0, 0], port_std[0, 0]])

    return weights_list, mean_std_list



port_weight, port_mean_std = minimum_port_weight(etfs_ret.values[:,1:])


np.shape(port_mean_std)