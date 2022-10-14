import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,precision_score,plot_roc_curve,accuracy_score,recall_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier


"""
Start
"""

# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?datasetId=310&sortBy=dateRun&tab=profile
# 492 frauds / 284807 transactions -> 473 frauds / 283726 transactions
# normal : Class = 0 , fraud : Class = 1
df = pd.read_csv('C://data_minsung/kaggle/creditcard/creditcard.csv')

"""
Data EDA(Exploratory Data Analysis)
"""
# check data information(data type, null check, # of observations)
df.info()

# check duplication -> 1081
# remove duplication 284808 -> 283726
df.duplicated().sum()
df = df.drop_duplicates()

# countplot
sns.countplot(x='Class', data=df, palette='pastel')
df.Class.sum()

# pie plot(needs to install plotly)
f_or_n = df["Class"].value_counts().tolist()
# values = [f_or_n[0], f_or_n[1]]
# fig = px.pie(values=data['Class'].value_counts(), names=lis , width=800,
#              height=400, color_discrete_sequence=["skyblue","black"]
#              ,title="percentage between Frauds & genuin transactions")
# fig.show()


# correlation heatmap
corr = df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11,9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.5, center=0, linewidth=.5, cbar_kws={"shrink":.5})

# variable information check
plt.figure(figsize=(6,4))
sns.kdeplot(data=df[df['Class'] == 0]['V20'], label="Normal", shade=True)
sns.kdeplot(data=df[df['Class'] == 1]['V20'], label="Fraud", shade=True)
plt.legend()
plt.show()



# data scaling
std_scaler = StandardScaler()
df['Scaled_Amount'] = std_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df.drop(['Amount', 'Time'], axis=1, inplace=True)

X = df.drop('Class', axis=1)
y = df['Class']

# visualize with tsne (집 컴퓨터로 돌리기)
# X_tsne = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(X)

# smote for imbalanced data
smote=SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=0)


"""
# Classifier1 - Gaussian Naive Bayes
"""
classifier = GaussianNB()
classifier.fit(X_train, y_train)
 
# mean accuracy
classifier.score(X_train, y_train).round(5)
classifier.score(X_test, y_test).round(5)

naive_preds = classifier.predict(X_test)
conf_mat = confusion_matrix(y_test, naive_preds)

plot_confusion_matrix(classifier, X_test, y_test)
plt.title("Naive : Confusion Matrix", fontsize=14)
plt.show()

accuracy_score(y_test, naive_preds)
recall_score(y_test, naive_preds)
precision_score(y_test, naive_preds)
f1_score(y_test, naive_preds)

"""
# Classifier2 - Decision Tree
"""

dt = DecisionTreeClassifier(max_features=8, max_depth=6)
dt.fit(X_train, y_train)

dt.score(X_test, y_test).round(5)

DT_preds = dt.predict(X_test)
confusion_matrix(y_test, dt_preds)

# confusion matrix plot
plot_confusion_matrix(dt, X_test, y_test )
plt.title("Confusion Matrix : Decision Tree", fontsize=14)
plt.show()

print(f'\t\tDT Model has A:- \n\nAccuracy: {accuracy_score(y_test,DT_preds).round(4)}'
      f'\t\trecall_Score: {recall_score(y_test,DT_preds).round(4)}'
      f'\nPrecision_score: {precision_score(y_test,DT_preds).round(4)}'
      f'\t\tF1-score equals: {f1_score(y_test,DT_preds).round(4)}')


from sklearn.metrics import classification_report
print(classification_report(y_test, DT_preds))