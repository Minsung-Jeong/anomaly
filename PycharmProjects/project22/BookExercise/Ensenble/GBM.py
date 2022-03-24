# Boosting : weak learner를 여러 개 이용해서 학습에 이용(약한 분류기 보통 decision tree)
# 1. AdaBoost : 오류 데이터에 가중치를 부여하면서 부스팅, 2.GBM : Gradient descent를 이용해서 업데이트
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import time
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

# GBM 수행 시간 측정을 위한 시작시간 설정
start_time = time.time()
gb_clf = GradientBoostingClassifier(random_state=0)
gb_clf.fit(X_train, y_train)
gb_pred = gb_clf.predict(X_test)
gb_acc = accuracy_score(y_test, gb_pred)
print("acc:{0:4f}, time:{1}".format(gb_acc, time.time()-start_time))


# grid search 진행
params = {'n_estimators':[100,500], 'learning_rate':[0.05, 0.01]}
grid_cv = GridSearchCV(gb_clf, param_grid=params, cv=2, verbose=1)
grid_cv.fit(X_train, y_train)
print('최적 hyperparam\n',grid_cv.best_params_)
print('최적 score:{0:4f}'.format(grid_cv.best_score_))