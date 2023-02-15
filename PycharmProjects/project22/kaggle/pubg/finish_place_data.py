import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import time
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

# simple shot
def simple_shot(var):
    print('mean value of {} : {}'.format(var, train[var].mean()))
    print('median value of {} : {}'.format(var, train[var].median()))
    print('top 1% of {} : {}'.format(var, train[var].quantile(0.99)))
    print('max of {} : {}'.format(var, train[var].max()))
    print('min of {} : {}'.format(var, train[var].min()))

# -------------------------------------kill
# Killer에 대한 정보(평균적으로 1킬을 못함, top 1%는 7킬, 최대킬은 72킬)
simple_shot('kills')



# 킬과 승리의 상관관계(0.4199로 높지 않음)
sns.jointplot(x="winPlacePerc", y="kills", data=train, height=10, ratio=3, color="r")
plt.show()
kill_win_corr = train[["winPlacePerc", "kills"]].corr()

# 킬과 승리에 대한 scatter plot
simple_shot('rankPoints')
plt.hist(train['rankPoints'])
plt.scatter(round(train['rankPoints'].median()), 1500000, c='red')
plt.show()

def rank_split(var):
    temp = []
    crit1 = var.quantile(0.2)
    crit2 = var.quantile(0.4)
    crit3 = var.quantile(0.6)
    crit4 = var.quantile(0.8)
    for i in range(len(train)):
        if var[i] <= crit1:
            temp.append(1)
        elif crit1 < var[i] <= crit2:
            temp.append(2)
        elif crit2 < var[i] <= crit3:
            temp.append(3)
        elif crit3 < var[i] <= crit4:
            temp.append(4)
        elif crit4 < var[i]:
            temp.append(5)
    return temp


abc = temp[4].copy()
abc = abc[abc.values != 0]

def split_corr(input_val):
    temp = input_val.stack()
    temp.index = input_val.index
    li = []
    for i in range(1, 6):

        data = temp[i]
        data = data[data.values !=0]
        val = data.values
        idx = data.index
        # val = temp[i].values
        # idx = temp[i].index

        correl = pd.DataFrame([idx, val]).T.corr()
        res = correl[0][1]
        li.append(res)
        print("rank {}'s correlation :{}".format(i, res))
    return li

# 랭크 내에서 킬에 따른 승리확률 시각화
train['rankPoints_split'] = rank_split(train['rankPoints'])
rank_kill_win = train[['winPlacePerc', 'rankPoints_split', 'kills']].groupby(['rankPoints_split', 'kills']).mean()

ax = rank_kill_win.unstack(level=0).plot(kind='bar', subplots=True, rot=0, figsize=(9, 7), layout=(2, 3))
plt.tight_layout()


# 결론 : 높은 랭크에서 킬과 승리의 상관관계가 높은편, 하위 랭크에서 많이 낮음
rank_kill_win_corr = split_corr(rank_kill_win)

temp = rank_kill_win.stack()
temp.index = rank_kill_win.index


# 팀킬에 대해 알아보자
simple_shot('teamKills')
# 팀킬러는 거의 없다(97.8퍼센트가 팀킬0)
sns.displot(train['teamKills'])
plt.show()

# 팀킬 0, 보다 팀킬 한 사람 승률높음(innocent, troller의 데이터 불균형 심함 or bad guys are prone to survive)
innocent = train[train['teamKills'] == 0]['winPlacePerc'].mean()
troller = train[train['teamKills'] !=0]['winPlacePerc'].mean()


# ----------------------------walkDistance
# 존버 승리자의 비율 - kill 낮으면서, walkDistance 높은 사람의 승리확률
# 걷는거리 랭크 내에서 킬과 승리의 상관관계
# 4,5번 split 보면 분석결과 마라토너는 킬을 적게 한다. 최상위 마라토너들은 솔로가 아니라 듀오 또는 스쿼드

# 심플샷
simple_shot('walkDistance')

# 데이터분포 - 이동거리 짧은 사람이 많다 = 시작하자 마자 죽는 사람이 많다 = 개선가능한 사항?
data = train.copy()
data = data[data['walkDistance'] < train['walkDistance'].quantile(0.99)]
sns.displot(data['walkDistance'])
plt.show()

# x : 승리, y : 걸은 거리
# sns.jointplot(x="walkDistance", y="winPlacePerc",  data=train, height=10, ratio=3, color="lime")
sns.jointplot(x="winPlacePerc", y="walkDistance",  data=train, height=10, ratio=3, color="lime")
plt.show()

# 걷는 거리에 대한 rank_split
train['walk_split'] = rank_split(train['walkDistance'])

# 랭크 별 걷기와 승리의 상관 : 모든 랭크에서 걷기와 승리는 높은 상관성을 보임
rank_walk_win = train[['winPlacePerc', 'walk_split', 'rankPoints_split']].groupby(['rankPoints_split', 'walk_split']).mean()
ax = rank_walk_win.unstack(level=0).plot(kind='bar', subplots=True, rot=0, figsize=(9, 7), layout=(2, 3))
plt.tight_layout()
walk_kill_win_corr = split_corr(rank_walk_win) #많이 걷는 그룹에서 킬과 승리의 상관성이 낮게 나옴


# 걷기 레벨 내에서 킬과 승리의 상관성
walk_kill_win = train[['winPlacePerc', 'walk_split', 'kills']].groupby(['walk_split', 'kills']).mean()
ax = walk_kill_win.unstack(level=0).plot(kind='bar', subplots=True, rot=0, figsize=(9, 7), layout=(2, 3))
plt.tight_layout() #plot보다 아래에서 상관계수 뽑는게 더 직관적
walk_kill_win_corr = split_corr(walk_kill_win) #많이 걷는 그룹에서 킬과 승리의 상관성이 낮게 나옴(많이 걷는 것이 영리?)


## Hypothesis : 장거리 마라토너로 승리하는 사람은 솔로가 아닐 확률이 높다? - 장거리 마라토너는 무임승차자가 많다
solos = train[train['numGroups']>50]
# duos = train[(train['numGroups']>25) & (train['numGroups']<=50)]
# squads = train[train['numGroups']<=25]
teamPlayers = train[train['numGroups'] <=50]

# solo or not으로 나눈 뒤 walkDistance로 나눈 뒤 win과 상관성
# 상관관계로만 봤을 때는 솔로나 팀이나 모두 walk - win 의 상관성 높음
sol_walk_win = solos[['walk_split', 'winPlacePerc']].groupby(['walk_split']).mean()
team_walk_win = teamPlayers[['walk_split', 'winPlacePerc']].groupby(['walk_split']).mean()

sol_walk_win_corr = pd.DataFrame([np.arange(1,6), np.squeeze(sol_walk_win.values)]).T.corr()[0][1]
team_walk_win_corr = pd.DataFrame([np.arange(1,6), np.squeeze(team_walk_win.values)]).T.corr()[0][1]

# solo or not으로 나눈뒤 walkDistance로 나누고 kill과 win의 상관성
sol_walk_kill_win = solos[['walk_split', 'winPlacePerc', 'kills']].groupby(['walk_split', 'kills']).mean()
team_walk_kill_win = teamPlayers[['walk_split', 'winPlacePerc', 'kills']].groupby(['walk_split', 'kills']).mean()

sol_walk_kill_win_corr = split_corr(sol_walk_kill_win) #솔로 : 걷기 랭킹 내에서 킬과 승리의 상관성이 높음
team_walk_kill_win_corr = split_corr(team_walk_kill_win) # 팀 : 걷기 랭킹 내에서 킬과 승리의 상관성이 낮음

# -------------
# temp = sol_walk_kill_win.stack()
# temp.index = sol_walk_kill_win.index
#
# data = temp[1]
# data = data[data.values!=0]
# ----------------

plt.title("solo vs team walk-kill-win correlation")
plt.plot(range(0,5), sol_walk_kill_win_corr, label='solo')
plt.plot(range(0,5), team_walk_kill_win_corr, label = 'team')
plt.legend()
plt.show()

# ---------------------------------boosts
# 심플샷

simple_shot('boosts')

# boosts에 대한 distrib
data = train.copy()
data = data[data['boosts'] < train['boosts'].quantile(0.99)]
sns.displot(data['boosts'])
plt.show()

# x : win, y: boosts 으로 한 jointplot
sns.jointplot(x="winPlacePerc", y="boosts",  data=train, height=10, ratio=3, color="lime")
plt.show()

# 부스트와 다른 변수간의 관계(1.walk - 2.win - 3.heal)
boost_corr = train.corr()["boosts"].sort_values(ascending=False)

# 랭크레벨 당 부스트와 승리
rank_boost_win = train[['winPlacePerc', 'rankPoints_split', 'boosts']].groupby(['rankPoints_split', 'boosts']).mean()


ax = rank_boost_win.unstack(level=0).plot(kind='bar', subplots=True, rot=0, figsize=(9, 7), layout=(2, 3))
plt.tight_layout()

# 랭크 내에서 부스트와 승리의 상관성 - 랭크 무관 부스트는 상관성이 높은편
rank_boost_win = split_corr(rank_boost_win)





# ------마지막
# 의문 heals은 0.4, boosts 는 0.6 Why?
