import pandas as pd
import numpy as np

dfmi = pd.DataFrame([list('abcd'),
                     list('efgh'),
                     list('ijkl'),
                     list('mnop')],
                    columns=pd.MultiIndex.from_product([['one', 'two'],
                                                        ['first', 'second']]))

dfmi['one']['second'].iloc[0] = 'a'
dfmi.loc[:, ('one', 'second')].iloc[0] = 'a'





os.chdir('C://data_minsung/finance/Qraft')
etfs = pd.read_csv('./indexPortfolio/etfs.csv')
etfs = etfs.set_index('Date')


# 총 nan = 5개/ min = 1829, max = 3413
etfs_ret = etfs.pct_change(1).iloc[1:]
etfs_ret = etfs_ret.iloc[:60].dropna(axis=1).T
asset_n = etfs_ret.index.values


mean_ret = np.mean(etfs_ret, axis=1)
cov_ret = np.cov(etfs_ret)
n = mean_ret.shape[0]
one_array = np.ones(n)
# ------------------------------

rf = 0.0001
mu = np.matrix(mean_ret).T
cov = np.matrix(cov_ret)
cov_inv = np.linalg.inv(cov)

numerator = np.dot(cov_inv, mu)
one_matrix = np.matrix(np.ones(mu.shape[0]))
denominator = np.dot(one_matrix, cov_inv).dot(mu)

tan_weight = numerator / denominator[0,0]
weight_df = pd.DataFrame(tan_weight, index=asset_n)

def re_weight(weight_df):
    if weight_df.values.min() < 0:
        weight_df = weight_df - weight_df.min()
    weight_df = weight_df/ weight_df.sum()
    weight_df = weight_df.sort_values(by=0, ascending=False)
    weight_val = weight_df.values
    for i in range(len(weight_val)):
        if weight_val[i] > 0.25 and i != len(weight_val)-1:
            diff = weight_df.iloc[i] - 0.25
            weight_df.iloc[i + 1] = weight_df.iloc[i + 1] + diff
            weight_df.iloc[i] = 0.25
        if weight_val[i] > 0.25 and i == len(weight_val) - 1:
            weight_df.iloc[i] = 0.25
    temp = []
    for A in asset_n:
        temp.append(weight_df.T[A].values[0])
    weight_df = pd.DataFrame(temp, index=asset_n)
    return weight_df

re_weighted = re_weight(weight_df)

