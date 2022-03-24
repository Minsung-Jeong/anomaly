import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
feature_name_df = pd.read_csv('C://data_minsung/human_act/features.txt', sep='\s+', header=None, names=['column_index', 'column_name'])

feature_name = feature_name_df.iloc[:,1].values.tolist()

len(feature_name)
feature_name_df

feature_dup_df = feature_name_df.groupby('column_name').count()
feature_dup_df[feature_dup_df['column_index']>1].count()

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

    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = get_human_dataset()

y_train['action'].value_counts()
y_test['action'].value_counts()

# decision tree 생성
dt_clf = DecisionTreeClassifier(random_state=156)
dt_clf.fit(X_train, y_train)
pred = dt_clf.predict(X_test)
accuracy = accuracy_score(y_test,pred)
print('decision tree 예측 정확도:{0:3f}'.format(accuracy))

params = {'max_depth':[6,8,10,12,16,20,24]}
grid_cv = GridSearchCV(dt_clf, param_grid=params, scoring='accuracy', cv=5, verbose=1)
grid_cv.fit(X_train, y_train)
print("최고:{0:4f}".format(grid_cv.best_score_))
print("최고param:",grid_cv.best_params_)


# 각 하이퍼파라미터의 성능 수치를 순서대로
cv_results_df = pd.DataFrame(grid_cv.cv_results_)
cv_results_df[['param_max_depth', 'mean_test_score']]

# with test data
for depth in params['max_depth']:
    dt_clf = DecisionTreeClassifier(max_depth=depth, random_state=156)
    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    print("max_dpt:{0} 정확도:{1:4f}".format(depth, accuracy))