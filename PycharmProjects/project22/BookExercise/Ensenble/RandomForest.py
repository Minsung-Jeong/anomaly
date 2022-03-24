# random Forest는 Bagging 의 일종
# bagging 은 같은 알고리즘으로 voting하는 것
# bagging = bootstrap aggregation(한 데이터 여러 개로 자르고 조합해서 여러 개의 같은 알고리즘에 넣고 voting)
import pandas as pd
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

def get_new_feature_name_df(old_feature_name_df):
    feature_dup_df = pd.DataFrame(data=old_feature_name_df.groupby('column_name').cumcount(), columns=['dup_cnt'])
    feature_dup_df = feature_dup_df.reset_index()
    new_feature_name_df = pd.merge(old_feature_name_df.reset_index(), feature_dup_df, how='outer')
    new_feature_name_df['column_name'] = new_feature_name_df[['column_name', 'dup_cnt']].apply(lambda x: x[0]+'_'+str(x[1])
                                                                                               if x[1]>0 else x[0], axis=1)
    new_feature_name_df = new_feature_name_df.drop(['dup_cnt'], axis=1)
    return new_feature_name_df

def get_human_dataset():

    feature_name_df = pd.read_csv('C://data_minsung/human_act/features.txt', sep='\s+', header=None, names=['column_index', 'column_name'])
    new_feature_name_df = get_new_feature_name_df(feature_name_df)
    feature_name = new_feature_name_df.iloc[:,1].values.tolist()

    X_train = pd.read_csv('C://data_minsung/human_act/train/X_train.txt', sep='\s+', names = feature_name)
    X_test = pd.read_csv('C://data_minsung/human_act/test/X_test.txt', sep='\s+', names = feature_name)

    y_train = pd.read_csv('C://data_minsung/human_act/train/y_train.txt', sep='\s+', header=None, names = ['action'])
    y_test = pd.read_csv('C://data_minsung/human_act/test/y_test.txt', sep='\s+', header=None, names = ['action'])

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = get_human_dataset()

# random forest 쌩으로 실행
rf_clf = RandomForestClassifier(random_state=0)
rf_clf.fit(X_train, y_train)
pred = rf_clf.predict(X_test)
accuracy_score(y_test, pred)

# grid search 로 하이퍼파라미터 세팅하기
params = {'n_estimators':[100, 300],
          'max_depth':[6,8,10,12],
          'min_samples_leaf':[8,12,18],
          'min_samples_split':[8,16,20]}

rt_clf = RandomForestClassifier(random_state=0, n_jobs=1)
grid_cv = GridSearchCV(rt_clf, param_grid=params, cv=2, n_jobs=1)
grid_cv.fit(X_train, y_train)

print("최적 하이퍼 파라미터:\n", grid_cv.best_params_)
print("best score:{0:.4f}".format(grid_cv.best_score_))



# estimator = 300으로 만들어보기
# rf_clf1 = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=8, min_samples_split=8)
rf_clf1 = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=8, min_samples_split=8)

rf_clf1.fit(X_train, y_train)
pred = rf_clf1.predict(X_test)
print("acc:{0:4f}".format(accuracy_score(y_test, pred)))

# 중요도 순 상위 20개의 features
ftr_importances_values = rf_clf1.feature_importances_
ftr_importances = pd.Series(ftr_importances_values, index=X_train.columns)
ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]

plt.figure(figsize=(8,6))
plt.title('feature importances Top 20')
sns.barplot(y=ftr_top20, x=ftr_top20.index)
plt.show()

# feature importance 뽑아내기
