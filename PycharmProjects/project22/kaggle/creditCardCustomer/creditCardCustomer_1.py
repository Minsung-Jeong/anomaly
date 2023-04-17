import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C://data_minsung/kaggle/creditcardCustomer/BankChurners.csv")

# ignore it
df = df.iloc[:,:-2]


df.head()
df.info()


# fig, ax = plt.subplots()
# ax.boxplot([df['Customer_Age']])
# # ax.set_ylim(-10.0, 10.0)
# ax.set_xlabel('customer_age')
# ax.set_ylabel('y_val')
# plt.show()

df.columns

df_copy = df.copy()
df_copy = df_copy.drop(['Attrition_Flag','Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category'], axis=1)

plt.figure(figsize=(15,8))
plt.title('heatmap')
sns.heatmap(df_copy, annot=True, fmt=".1f",vmin = 0.1,cmap="BuPu_r")
plt.show()