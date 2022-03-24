import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

raw_data = pd.read_csv("C://data_minsung/pima_Indians_diabetes/diabetes.csv")


for x in raw_data.columns:
    print(sum(raw_data.loc[:,x]))

# 데이터 살펴보기
# 음성 : 500, 양성 268, 의료데이터이기 때문에 recall 중점적으로 살펴보기(TPR, 실제양성을 양성으로 말하기)
raw_data.Outcome.value_counts()
raw_data.info()
raw_data.describe()

# 데이터 군에서 0이 얼마나 있는지 찾기
zero_features = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
total_count = len(raw_data)

for feature in zero_features:
    zero_count = raw_data[raw_data[feature] == 0][feature].count()
    print('{0} 0 건수는 {1}, 퍼센트는 {2:.2f}%'.format(feature, zero_count, zero_count/total_count*100))

# 0을 평균값으로 대체
mean_zero_features = raw_data[zero_features].mean()
raw_data[zero_features] = raw_data[zero_features].replace(0, mean_zero_features) #파이썬은 이런 식으로 데이터 퉁칠 수 있는 것 기억, 복습하기



X = raw_data.iloc[:,:-1]
Y = raw_data.iloc[:,-1]

scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, train_size=0.8, random_state=78)



lr_clf = LogisticRegression()
lr_clf.fit(X_train, Y_train)
pred = lr_clf.predict(X_test)
pred_prob = lr_clf.predict_proba(X_test)
cf_matrix = confusion_matrix(Y_test, pred)

TN = cf_matrix[0,0]
FP = cf_matrix[0,1]
FN = cf_matrix[1,0]
TP = cf_matrix[1,1]

accuracy = accuracy_score(Y_test, pred)
precision = precision_score(Y_test, pred)
recall = recall_score(Y_test, pred)

f1_score = 2*(precision*recall)/(precision+recall)

fprs, tprs, thresholds = roc_curve(Y_test, pred_prob[:,1])
roc_score = roc_auc_score(Y_test, pred_prob[:,1])

plt.plot(fprs, tprs, label="ROC")
plt.plot([0,1],[0,1], 'k--', label="Random")
# thr_idx = np.arange(1, thresholds.shape[0], 5)
# pred_prob_c1 = lr_clf.predict_proba(X_test)[:,1]

