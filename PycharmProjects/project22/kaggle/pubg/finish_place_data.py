import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

train = pd.read_csv("C://data_minsung/kaggle/pubg/train2.csv")

train.info()
train.isnull().sum()
train.head()

# 모든 변수들의 상관관계 살펴보기(상위 3개 변수에 대해서 분석 + 의외의 변수 1개(kills) => 4개의 변수 분석)
f, ax = plt.subplots(figsize=(20,20))
sns.heatmap(train.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
plt.show()

# walkDistance, boosts, weaponsAcquired, kills
corr_rank = train.corr()["winPlacePerc"].sort_values(ascending=False)

# 해당 게임은 실력이 비슷한 사람끼리 매칭됨
# killpoints, rankpoints, winpoints 와 winPlacePerc는 상관성 낮음 -> 매칭시스템이 잘 잡혀 있다는 의미로 해석
# 그것말고는 당장 의미를 찾을 부분은 없을 듯
# sns.jointplot(x="rankPoints", y="winPlacePerc", data=train, height=10, ratio=3, color="lime")
# train[['rankPoints', 'winPlacePerc']].corr()
#
# sns.jointplot(x="winPoints", y="winPlacePerc", data=train, height=10, ratio=3, color="lime")
# train[['winPoints', 'winPlacePerc']].corr()
#
# sns.jointplot(x="killPoints", y="winPlacePerc", data=train, height=10, ratio=3, color="lime")
# train[['killPoints', 'winPlacePerc']].corr()


# simple shot
def simple_shot(var):
    print('mean value of {} : {}'.format(var, train[var].mean()))
    print('median value of {} : {}'.format(var, train[var].median()))
    print('top 1% of {} : {}'.format(var, train[var].quantile(0.99)))
    print('max of {} : {}'.format(var, train[var].max()))

# -------------------------------------kill
# Killer에 대한 정보(평균적으로 1킬을 못함, top 1%는 7킬, 최대킬은 72킬)
simple_shot('kills')

# kill를 구분화해서 countplot
# data = train.copy()
# data.loc[data['kills'] > data['kills'].quantile(0.99)] = '8+'
# plt.figure(figsize=(15, 10))
# sns.countplot(data['kills'].astype('str').sort_values())
# plt.title("Kill Count", fontsize=15)
# plt.show()

# 킬과 승리의 상관관계(0.4199로 높지 않음)
sns.jointplot(x="winPlacePerc", y="kills", data=train, height=10, ratio=3, color="r")
plt.show()
kill_win_corr = train[["winPlacePerc", "kills"]].corr()

# 같은 랭킹 내에서 킬과 'winPlacePerc'
# 랭킹을 5단계로 구분
simple_shot('rankPoints')
plt.hist(train['rankPoints'])
plt.scatter(round(train['rankPoints'].median()), 1500000, c='red')
plt.show()

# split rankPoints into 5 levels
# -1에서 5910까지를 나누어 다섯 개의 랭크 생성
def rank_split(var_name):
    criteria = math.trunc((train[var_name].max() - train[var_name].min()) / 5)
    temp = []
    for i in range(len(train)):
        if train[var_name][i] < criteria:
            temp.append(1)
        elif criteria <= train[var_name][i] < criteria * 2:
            temp.append(2)
        elif criteria * 2 <= train[var_name][i] < criteria * 3:
            temp.append(3)
        elif criteria * 3 <= train[var_name][i] < criteria * 4:
            temp.append(4)
        elif criteria * 4 <= train[var_name][i]:
            temp.append(5)
    return temp

train['rankPoints_split'] = rank_split('rankPoints')
# rank_kill = train[['winPlacePerc', 'rankLevel', 'kills']].groupby(['rankLevel', 'kills']).mean()
# train.to_csv("C://data_minsung/kaggle/pubg/train2.csv")

# 랭크 내에서 킬에 따른 치킨확률 시각화
rank_kill = train[['winPlacePerc', 'rankPoints_split', 'kills']].groupby(['rankPoints_split', 'kills']).mean()
ax = rank_kill.unstack(level=0).plot(kind='bar', subplots=True, rot=0, figsize=(9, 7), layout=(2, 3))
plt.tight_layout()


temp = rank_kill.stack()
temp.index = rank_kill.index

# 랭크 포인트가 높을 수록 킬과 승리의 상관관계는 높다
for i in range(1, 6):
    val = temp[i].values
    idx = temp[i].index
    correl = pd.DataFrame([idx, val]).T.corr()
    res = correl[0][1]
    print("rank {}'s correlation :{}".format(i, res))



# ----------------------------walkDistance
# 존버 승리자의 비율 - kill 낮으면서, walkDistance 높은 사람의 승리확률
# 많이 움직일 수록 싸우지 않고 이길 수 있다
# 4,5번 split 보면 분석결과 마라토너는 킬을 적게 한다. 최상위 마라토너들은 솔로가 아니라 듀오 또는 스쿼드

simple_shot('walkDistance')

train['walk_split'] = rank_split('walkDistance')
walk_kill_win = train[['winPlacePerc', 'walk_split', 'kills']].groupby(['walk_split', 'kills']).mean()
ax = walk_kill_win.unstack(level=0).plot(kind='bar', subplots=True, rot=0, figsize=(9, 7), layout=(2, 3))
plt.tight_layout()

# rank_walk_win = train[['winPlacePerc', 'rankPoints_split', 'walk_split']].groupby(['rankPoints_split', 'walk_split']).mean()
# temp =train[['rankPoints_split', 'walk_split']].groupby(['rankPoints_split']).mean()

