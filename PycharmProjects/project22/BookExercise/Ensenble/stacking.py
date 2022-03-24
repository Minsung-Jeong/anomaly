import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# stacking 은 마지막 영끌하며 정확도 올리는 느낌
# 따라서 과적합의 가능성 있음
cancer_data = load_breast_cancer()

cancer_data
X_data = cancer_data.data
y_label = cancer_data.target

X_train, X_test, y_train, y_test = train_test_split(X_data, y_label, test_size=0.2, random_state=0)

# 개별모델
knn_clf = KNeighborsClassifier(n_neighbors=4).fit(X_train, y_train)
rf_clf = RandomForestClassifier(n_estimators=100, random_state=0).fit(X_train, y_train)
dt_clf = DecisionTreeClassifier().fit(X_train, y_train)
ada_clf = AdaBoostClassifier(n_estimators=100).fit(X_train, y_train)


# 최종예측 모델
lr_clf = LogisticRegression(C=10)

# 개별모델 학습
knn_pred = knn_clf.predict(X_test)
rf_pred = rf_clf.predict(X_test)
dt_pred = dt_clf.predict(X_test)
ada_pred = ada_clf.predict(X_test)


print('knn정확도:{0:4f}'.format(accuracy_score(y_test, knn_pred)))
print('random forest정확도:{0:4f}'.format(accuracy_score(y_test, rf_pred)))
print('decision tree정확도:{0:4f}'.format(accuracy_score(y_test, dt_pred)))
print('ada boost정확도:{0:4f}'.format(accuracy_score(y_test, ada_pred)))

pred = np.array([knn_pred, rf_pred, dt_pred, ada_pred])
pred = np.transpose(pred)

lr_clf.fit(pred, y_test)
final_pred = lr_clf.predict(X_test)

# cross-validation 통해서 메타 clf에서 학습/테스트 둘 다 할 수 있게 하기
def get_stacking_base_datasets(model, X_train_n, y_train_n, X_test_n, n_folds):
    kf = KFold(n_splits=n_folds, shuffle=False, random_state=0)

    train_fold_pred = np.zeros((X_train_n.shape[0], 1))
    test_pred = np.zeros((X_test_n.shape[0], n_folds))

    for folder_counter, (train_index, valid_index) in enumerate(kf.split(X_train_n)):
        print('\t 폴드 세트: ', folder_counter, ' 시작')
        # print('train idx:{0}, valid idx:{1}'.format(train_index, valid_index))
        X_tr = X_train_n[train_index]
        y_tr = y_train_n[train_index]
        X_te = X_train_n[valid_index]
        # print('x_tr size:{0}, x_te size:{1}'.format(np.shape(X_tr), np.shape(X_te)))
        model.fit(X_tr, y_tr)
        train_fold_pred[valid_index, :] = model.predict(X_te).reshape(-1, 1)
        test_pred[:, folder_counter] = model.predict(X_test_n)

    test_pred_mean = np.mean(test_pred, axis=1).reshape(-1, 1)

    return train_fold_pred, test_pred_mean

knn_train, knn_test = get_stacking_base_datasets(knn_clf, X_train, y_train, X_test, 7)
rf_train, rf_test = get_stacking_base_datasets(rf_clf, X_train, y_train, X_test, 7)
dt_train, dt_test = get_stacking_base_datasets(dt_clf, X_train, y_train, X_test, 7)
ada_train, ada_test = get_stacking_base_datasets(ada_clf, X_train, y_train, X_test, 7)

Stack_final_X_train = np.concatenate((knn_train, rf_train, dt_train, ada_train), axis=1)
Stack_final_X_test = np.concatenate((knn_test, rf_test, dt_test, ada_test), axis=1)

lr_clf.fit(Stack_final_X_train, y_train)
stack_final = lr_clf.predict(Stack_final_X_test)
accuracy_score(y_test, stack_final)