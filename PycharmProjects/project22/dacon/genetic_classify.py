import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from catboost import CatBoostClassifier

import catboost as cb
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

import os
from sklearn.preprocessing import LabelEncoder

def get_dict(data):
    dic_x = {}
    for i, x in enumerate((data), start=0):
        dic_x[x] = i
    return dic_x

os.getcwd()
os.chdir("C://data_minsung/dacon")
train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')

# father, mother, gender 필요 없는 데이터
# sum(train_df['father'])
# sum(test_df['father'])
# sum(train_df['mother'])
# sum(test_df['mother'])
# sum(train_df['gender'])
# sum(test_df['gender'])

# train, test 데이터 같음
snp_col = train_df.columns[5:-1]
for col in snp_col:
    dic = get_dict(list(set(train_df[col])))
    for i in range(len(train_df)):
        train_df[col][i] = dic[train_df[col][i]]
    for i in range(len(test_df)):
        test_df[col][i] = dic[test_df[col][i]]




train_df = train_df.drop(['id','father','mother', 'gender'], axis=1)
test_df = test_df.drop(['id','father','mother', 'gender'], axis=1)

le = LabelEncoder()  # from sklearn.preprocessing


# label_dic = get_dict(list(set(train_df['class'])))
# train_y = train_df['class'].copy()
# for i in range(len(train_y)):
#     train_y[i] = label_dic[train_y[i]]
train_y = le.fit_transform(train_df['class'].copy())


train_x = train_df.drop('class', axis=1)

# train_df.groupby(by='class').count()


# Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits=10)

# Modeling step Test differents algorithms
random_state = 2
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(LinearDiscriminantAnalysis())


cv_results = []
for classifier in classifiers:
    cv_results.append(cross_val_score(classifier, train_x, y = le.fit_transform(train_y), scoring = "accuracy", cv = kfold, n_jobs=4))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",
"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})

# g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
# g.set_xlabel("Mean Accuracy")
# g = g.set_title("Cross validation scores")

cv_res.sort_values(by='CrossValerrors')


################Hyperparameter tunning
############################## 1. Adaboost
DTC = DecisionTreeClassifier()
adaDTC = AdaBoostClassifier(DTC, random_state=7)
ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[1,2],
              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}

gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsadaDTC.fit(train_x,train_y)
ada_best = gsadaDTC.best_estimator_
print(gsadaDTC.best_score_)

########################### 2. ExtraTrees
ExtC = ExtraTreesClassifier()
## Search grid for optimal parameters
ex_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}
gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsExtC.fit(train_x,train_y)
ExtC_best = gsExtC.best_estimator_
# Best score
gsExtC.best_score_

################## 3. Random Forest
RFC = RandomForestClassifier()
## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}

gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsRFC.fit(train_x, train_y)
RFC_best = gsRFC.best_estimator_
# Best score
gsRFC.best_score_

######################### 4. Gradient boosting tunning(성능 안 좋음) - acc 77나옴
GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1]
              }
gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsGBC.fit(train_x,train_y)
GBC_best = gsGBC.best_estimator_
# Best score
gsGBC.best_score_

#################### 5. SVC classifier
SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'],
                  'gamma': [0.0001, 0.001, 0.01, 0.1],
                  'C': [50, 100,200,300, 1000]}
gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsSVMC.fit(train_x,train_y)
SVMC_best = gsSVMC.best_estimator_
# Best score
gsSVMC.best_score_

##################### 6. MLP Classifier
MLP = MLPClassifier()
mlp_param_grid = {'max_iter' : [300,400,500],
                  'hidden_layer_sizes' : [32, 64, 128, 256],
                    'alpha': 10.0 ** -np.arange(3, 8)
                  }

gsMLP = GridSearchCV(MLP, param_grid=mlp_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)
gsMLP.fit(train_x, train_y)
MLP_best = gsMLP.best_estimator_
gsMLP.best_score_

##################### 7. Logistic Regression
LR = LogisticRegression()
lr_param_grid = {"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}
gsLR = GridSearchCV(LR, param_grid=lr_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)
gsLR.fit(train_x,train_y)
LR_best = gsLR.best_estimator_
gsLR.best_score_

##################### 8. catboost
CB = CatBoostClassifier()
parameters = {'depth': [4, 5, 6, 7, 8, 9, 10],
              'learning_rate': [0.005, 0.001, 0.0001],
              'iterations': [30,  70,  100]
              }
gsCB= GridSearchCV(CB, param_grid=TEMP, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)
gsCB.fit(train_x, train_y)
CB_best = gsCB.best_estimator_
gsCB.best_score_ # 0.9541 > 0.92(depth 종류 적을 때)

"""
결과 시각화 통해서 확인
"""
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

g = plot_learning_curve(gsRFC.best_estimator_,"RF mearning curves",train_x,train_y,cv=kfold)
g = plot_learning_curve(gsExtC.best_estimator_,"ExtraTrees learning curves",train_x,train_y,cv=kfold)
g = plot_learning_curve(gsSVMC.best_estimator_,"SVC learning curves",train_x,train_y,cv=kfold)
g = plot_learning_curve(gsadaDTC.best_estimator_,"AdaBoost learning curves",train_x,train_y,cv=kfold)
g = plot_learning_curve(gsGBC.best_estimator_,"GradientBoosting learning curves",train_x,train_y,cv=kfold)
g = plot_learning_curve(gsMLP.best_estimator_,"MLP learning curves", train_x, train_y, cv=kfold)

nrows = ncols = 2
fig, axes = plt.subplots(nrows = nrows, ncols = ncols, sharex="all", figsize=(15,15))

names_classifiers = [("AdaBoosting", ada_best),("ExtraTrees",ExtC_best),("RandomForest",RFC_best),("GradientBoosting",GBC_best)]

nclassifier = 0
for row in range(nrows):
    for col in range(ncols):
        name = names_classifiers[nclassifier][0]
        classifier = names_classifiers[nclassifier][1]
        indices = np.argsort(classifier.feature_importances_)[::-1][:40]
        g = sns.barplot(y=train_x.columns[indices][:40],x = classifier.feature_importances_[indices][:40] , orient='h',ax=axes[row][col])
        g.set_xlabel("Relative importance",fontsize=12)
        g.set_ylabel("Features",fontsize=12)
        g.tick_params(labelsize=9)
        g.set_title(name + " feature importance")
        nclassifier += 1

"""
ensemble modeling
"""
#
# 첫번째 시도 - model_result_0, 0.96223, 76등
# votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),
# ('svc', SVMC_best), ('adac',ada_best),('gbc',GBC_best)], voting='soft', n_jobs=4)
# votingC = votingC.fit(train_x, train_y)

# # 두번째 시도 - 성능 안 좋은 GBC 제거 후 앙상블, 0.96223
# votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),
# ('svc', SVMC_best), ('adac',ada_best)], voting='soft', n_jobs=4)
# votingC = votingC.fit(train_x, train_y)

# # 세번째 시도 - 두 번째에 mlp 추가후 앙상블, 0.95237
# votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),
# ('svc', SVMC_best), ('adac',ada_best), ('mlp', MLP_best)], voting='soft', n_jobs=4)

# 네번째 시도 - GBC, MLP 다 추가,  0.95237
# votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),
# ('svc', SVMC_best), ('adac',ada_best), ('mlp', MLP_best) ,('gbc',GBC_best)], voting='soft', n_jobs=4)
# votingC = votingC.fit(train_x, train_y)

# # 다섯번째 - 이전에서 ada, gbc 제거 후 진행, 0.96223
# votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),
# ('svc', SVMC_best), ('mlp', MLP_best) ], voting='soft', n_jobs=4)
# votingC = votingC.fit(train_x, train_y)

#  여섯번쩨 - 위에서 mlp 제거, 0.95237
# votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),
# ('svc', SVMC_best)], voting='soft', n_jobs=4)
# votingC = votingC.fit(train_x, train_y)

# 일곱번쩊 - 첫번째에 logistic 추가
# votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),
# ('svc', SVMC_best), ('adac',ada_best),('gbc',GBC_best), ('lr', LR_best)], voting='soft', n_jobs=4)
# votingC = votingC.fit(train_x, train_y)

# # 여덟번째 - 첫번째에 logistic, mlp 추가
# votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),
# ('svc', SVMC_best), ('adac',ada_best), ('lr', LR_best), ('mlp', MLP_best)], voting='soft', n_jobs=4)
# votingC = votingC.fit(train_x, train_y)

# # 아홉번째 - 첫번째에 logistic, mlp, gbc 추가
# votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),
# ('svc', SVMC_best), ('adac',ada_best), ('lr', LR_best), ('mlp', MLP_best), ('gbc',GBC_best)], voting='soft', n_jobs=4)
# votingC = votingC.fit(train_x, train_y)

# # 열번째
votingC = VotingClassifier(estimators=[('extc', ExtC_best),
('svc', SVMC_best), ('mlp', MLP_best)], voting='soft', n_jobs=4)
votingC = votingC.fit(train_x, train_y)


pred = votingC.predict(test_df)

label_dic = get_dict(list(set(train_df['class'])))
label_rev_dic = {}
for i, x in enumerate(label_dic):
    label_rev_dic[i] = x

result = []
for x in pred:
    result.append(label_rev_dic[x])

temp = pd.read_csv('./test.csv')
temp['id']

result_df = pd.DataFrame(result, index=temp['id'], columns=['class'])
result_df.to_csv("./model_result_9.csv")

temp1 = pd.read_csv("./model_result_0.csv")
temp2 =  pd.read_csv("./model_result_2.csv")

sum(temp1['class'] != temp2['class'])
