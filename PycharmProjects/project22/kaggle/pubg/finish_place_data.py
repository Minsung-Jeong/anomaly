import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



train = pd.read_csv("C://data_minsung/kaggle/pubg/train_V2.csv")

train.info()
train.isnull().sum()
train.head()
# columns 정보
len(train.columns)

# 해당 게임은 실력이 비슷한 사람끼리 매칭됨
# killpoints, rankpoints, winpoints 와 winPlacePerc는 상관성 낮음 -> 매칭시스템이 잘 잡혀 있다는 의미로 해석
# 그것말고는 당장 의미를 찾을 부분은 없을 듯
sns.jointplot(x="rankPoints", y="winPlacePerc",  data=train, height=10, ratio=3, color="lime")
train[['rankPoints', 'winPlacePerc']].corr()

sns.jointplot(x="winPoints", y="winPlacePerc",  data=train, height=10, ratio=3, color="lime")
train[['winPoints', 'winPlacePerc']].corr()

sns.jointplot(x="killPoints", y="winPlacePerc",  data=train, height=10, ratio=3, color="lime")
train[['killPoints', 'winPlacePerc']].corr()

# simple shot
def simple_shot(var):
    print('mean value of {} : {}'.format(var, train[var].mean()))
    print('median value of {} : {}'.format(var, train[var].median()))
    print('top 1% of {} : {}'.format(var, train[var].quantile(0.99)))
    print('max of {} : {}'.format(var, train[var].max()))

# Killer에 대한 정보(평균적으로 1킬을 못함, top 1%는 7킬, 최대킬은 72킬)
simple_shot('kills')

# kill를 구분화해서 countplot
data = train.copy()
data.loc[data['kills'] > data['kills'].quantile(0.99)] = '8+'
plt.figure(figsize=(15,10))
sns.countplot(data['kills'].astype('str').sort_values())
plt.title("Kill Count",fontsize=15)
plt.show()

# 같은 랭킹 내에서 킬당 'winPlacePerc'
# 랭킹을 5단계로 구분
simple_shot('rankPoints')
plt.hist(train['rankPoints'])
plt.scatter(round(train['rankPoints'].median()), 1500000, c='red')

# sns.countplot(train['rankPoints'].astype('str').sort_values())
# train[['rankPoints','winPlacePerc']]