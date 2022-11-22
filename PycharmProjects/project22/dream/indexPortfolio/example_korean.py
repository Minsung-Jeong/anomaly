import yfinance as yf

import numpy as np
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import solvers
import pandas as pd



# Turn off progress printing(solvers의 진행 상태 표시 없애기)
solvers.options['show_progress'] = False

#Download price data from Yahoo finance (가격 데이터 다운로드)
p_apple = yf.download('AAPL',start = '2019-01-01')
p_google = yf.download('GOOGL',start = '2019-01-01')
p_amazon = yf.download('AMZN',start = '2019-01-01')

#Merge the tow tables above (표 병합)
p_apple = p_apple[['Adj Close']].rename(columns = {'Adj Close':'Close_Apple'})
p_google = p_google[['Adj Close']].rename(columns = {'Adj Close':'Close_Google'})
p_amazon = p_amazon[['Adj Close']].rename(columns = {'Adj Close':'Close_Amazon'})

price = pd.concat([p_apple,p_google,p_amazon],axis=1)

#Calculate the daily return of individual stock (개별 주식 일별 수익률 계산)
#daily returns = (today price - previous day price)/(previous day price) - 1
ptc_ret = price.pct_change(1).dropna()
ptc_ret = ptc_ret.rename(columns={'Close_Apple':'Apple','Close_Google':'Google','Close_Amazon':'Amazon'})

ret_matrix = ptc_ret.values.T

# Minimum variance portfolio (최소 분산 포트폴리오)
'''
min 1/2w'COVw
s.t. mu'w = rp and 1'w = 1
'''


def minimum_port_weight(returns):
    mean_ret = np.mean(returns, axis=1)
    cov_ret = np.cov(returns)
    n = mean_ret.shape[0]
    one_array = np.ones(n)

    # List of target portfolio returns (목표 포트폴리오 수익률 리스트)
    mus = list(np.arange(0, 0.003, 0.0001))

    # Convert to cvxopt matrices (cvxopt matrix로 변환)
    Q = opt.matrix(cov_ret)
    p = opt.matrix(0.0, (n, 1))

    # Create constraint matrices (최적화 문제 제약조건 설정)
    A = opt.matrix(np.stack((mean_ret, one_array)))
    weights_list = []
    mean_std_list = []
    for mu in mus:
        b = opt.matrix(np.array([mu, 1]))
        weights = solvers.qp(Q, p, A=A, b=b)['x']
        weight_array = np.array(list(weights))
        port_ret = np.matrix(weight_array).dot(np.matrix(mean_ret).T)
        port_std = np.sqrt(np.matrix(weight_array).dot(np.matrix(cov_ret)).dot(np.matrix(weight_array).T))

        weights_list.append(list(weights))
        mean_std_list.append([port_ret[0, 0], port_std[0, 0]])

    return weights_list, mean_std_list


weights,mean_stds = minimum_port_weight(ret_matrix)
#To dataframe (데이타 프레임으로 변경)
col_name = ptc_ret.columns + '_Weight'
weights_df = pd.DataFrame(weights,columns=col_name)
mean_stds_df = pd.DataFrame(mean_stds,columns=['Port_ret','Port_std'])