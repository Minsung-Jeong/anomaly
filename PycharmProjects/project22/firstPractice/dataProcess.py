import pandas as pd
import numpy as np
# from statsmodels.formula.api import ols
# import statsmodels.api as sm
import math
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

train_ = pd.read_csv("C://data_minsung/titanic/train.csv")
test_ = pd.read_csv("C://data_minsung/titanic/test.csv")

# column별 null값 찾기
train_.isnull().sum()


def mode(list):
    count = 0
    mode = 0
    for x in list:
        if list.count(x) > count:
            count = list.count(x)
            mode = x
    return mode

def pre_processing(train_, scale=False):
    # "Age"결측값 평균값으로 대체
    train_["Age"].fillna(round(train_["Age"].mean()), inplace=True)

    # "Sex" 의 male=0, female=1 로 대체
    train_.loc[train_.Sex == "male", "Sex"] = 0
    train_.loc[train_.Sex == "female", "Sex"] = 1

    # "Ticket은 일단 딕셔너리 생성해보기"
    ticket_set = list(set(train_.Ticket))
    ticket_dict = {}
    for i in range(len(ticket_set)):
        ticket_dict[ticket_set[i]] = i
    for i in range(len(train_.Ticket)):
        train_.loc[i ,"Ticket"] = ticket_dict[train_.Ticket[i]]

    # "Embarker"도 딕셔너리(결측치 제거 후)
    embark_set = list(set(train_.Embarked))
    train_.Embarked = train_["Embarked"].fillna(mode(embark_set))
    # 결측치 제거 후 다시 set
    embark_set = list(set(train_.Embarked))
    embark_dict = {}
    for i in range(len(embark_set)):
        embark_dict[embark_set[i]] = i

    for i, value in enumerate(train_.Embarked):
        train_.loc[i,"Embarked"] = embark_dict[value]

    # "Cabin", "Name", "PassengerId"은 삭제
    train_ = train_.drop(columns=["Cabin", "Name", "PassengerId"])

    # test는 "Survived" 없으므로 예외처리
    if scale:
        try:
            train_lb_drop = train_.drop("Survived", axis=1)
            scaler = StandardScaler()
            scaler.fit(train_lb_drop)
            scaled_train = scaler.transform(train_lb_drop)
            return scaled_train
        except:
            scaler = StandardScaler()
            scaler.fit(train_)
            scaled_train = scaler.transform(train_)
            return scaled_train
    else:
        train_ = train_.values[:,1:]

    return train_

train_data = pre_processing(train_)
test_data = pre_processing(test_, True)

train_data[9]
train_.iloc[9,:]

train_size = round(len(train_data)*0.8)
label = train_["Survived"].values

train_x = train_data[:train_size, :]
train_y = label[:train_size]

valid_x = train_data[train_size:, :]
valid_y = label[train_size:]

# 다변량 회귀(정규화 상관없이 83%)
mlr = LinearRegression()
mlr.fit(train_x, train_y)
regress_predict = np.round(mlr.predict(valid_x))
acc = sum(valid_y == regress_predict) / len(regress_predict)

# svm 모델(데이터 정규화 x 63%, 정규화 o 87%)
clf = svm.SVC()
clf.fit(train_x, train_y)
svm_predict = clf.predict(valid_x)
acc = sum(valid_y == svm_predict) / len(svm_predict)


cf_matrix = confusion_matrix(valid_y, regress_predict )
TN = cf_matrix[0,0]
FP = cf_matrix[0,1]
FN = cf_matrix[1,0]
TP = cf_matrix[1,1]

# 정확도(Accuracy)
acc = (TN+TP)/(len(svm_predict))
# 정밀도(precision) :양성으로 예측한 것 중 실제 양성 => 음성을 양성으로 할 시 큰 문제 - 스팸이 아닌데 스팸처리를 하는 경우
precision_m = TP / (FP + TP)
# 재현율( : 실제 양성 중 양성으로 예측한 률 => 양성을 음성으로 할 시 큰 문제 - 암환자인데 아니라고 하는 경우
recall_m = TP / (FN + TP)

accuracy = accuracy_score(valid_y, regress_predict)
precision = precision_score(valid_y, regress_predict)
recall = recall_score(valid_y, regress_predict)


# F1-score : precision 와 recall을 재결합
f1_score_m = 2*(precision*recall)/(precision+recall)

# TPR : 민감도, 재현율 = TP/(TP+FN)
# TNR : 특이성  = TN / (TN+FP)
# FPR =  FP / (FP+TN)  =  (1 - TNR)  =  (1 - 특이성)