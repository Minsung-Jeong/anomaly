import pandas as pd
import os
from sklearn.impute import KNNImputer
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt

os.chdir('C://data_minsung/kaggle/enzyme')
train_df = pd.read_csv("./train.csv")
train_extra = pd.read_csv("./train_updates_20220929.csv")
test_df = pd.read_csv("./test.csv")

train_df.head()
train_df.isnull().sum()
train_df.columns
# data source 는 test에서 동일한 것만 있으므로 무시
source_li = list(set(train_df['data_source'].values))
seq_li = list(set(train_df['protein_sequence'].values))
ph_li = list(set(train_df['pH'].values))
len(seq_li) / len(train_df) # seq의 종류가 observation 의 92%에 달하므로 단순 회귀 불가 - 자연어처리?

# sequences = train_df['protein_sequence'].values
# seq_value_len = [ len(s) for s in sequences]


# ph 데이터 보간
imputer = KNNImputer()
train_df['pH'] = imputer.fit_transform(np.expand_dims(train_df['pH'], axis=1))

# try1 : seq만으로 분석
# try2 : ph를 seq 마지막에 붙여서 같이 자연어처리 하듯이 진행
# try3 : seq, ph를 각자 다른 모델로 돌리고 앙상블

# ph와 tm간의 선형회귀 그려보기
# trn_size = int(len(train_df)*0.7)
def normalize(data):
    mean = data.mean()
    std = data.std()
    return (data-mean)/std


# regression plot 통해 tm과 ph간의 관계성이 크지 않음을 유추
Y = normalize(train_df['tm'])
X = normalize(train_df['pH'])

reg_data = pd.DataFrame()
reg_data['tm'] = Y.values
reg_data['ph'] = X.values

sns.regplot(x='ph', y='tm', data=reg_data)

