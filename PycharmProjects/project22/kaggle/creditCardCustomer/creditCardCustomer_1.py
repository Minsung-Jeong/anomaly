import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C://data_minsung/kaggle/creditcardCustomer/BankChurners.csv")

# ignore it
df = df.iloc[:,:-2]

df.head()
df.info()


def convert_num(variable):
    temp = list(set(df[variable]))
    temp_dict = {temp[i]: i for i in range(len(temp))}
    for x in temp_dict:
        df[variable][df[variable] == x] = temp_dict[x]
    return df[variable].astype(float), temp_dict

# 0 = existing Customer , 1 = attrited Customer
df['Attrition_Flag'], attrit_dict = convert_num('Attrition_Flag')
# 0 = female , 1 = male
df['Gender'], gender_dict = convert_num('Gender')
# 0=college, 1=graduate, 2=unknown, 3 = high school, 4= post_gradu, 5=doctor, 6=uneducated
df['Education_Level'], edu_dict = convert_num('Education_Level')
df['Marital_Status'], marital_dict = convert_num('Marital_Status')
df['Income_Category'], income_dict = convert_num('Income_Category')
df['Card_Category'], card_dict = convert_num('Card_Category')


# correlation heatmap
df_corr = df.corr()
plt.figure(figsize=(15,8))
plt.title('heatmap')
sns.heatmap(df_corr, annot=True, cmap="BuPu_r")
plt.show()

"""
# Correlation Top3 variables = Total_trans_Ct, Total_Ct_Chng_Q4_Q1, Total_Revolving_Bal
Total_Trans_Ct : total transaction count
Total_Ct_Chng_Q4_Q1 : Change in Transaction Count (Q4 over Q1)
Total_Revolving_Bal : Total Revolving Balance on the Credit Card
Contacts_Count_12_mon : No. of Contacts in the last 12 months
"""
abs(df_corr['Attrition_Flag']).sort_values(ascending=False)


# df[['Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Total_Revolving_Bal',  'Contacts_Count_12_mon']].plot.bar()

df_sample = df.sample(3000)
plt.bar(np.arange(len(df_sample)),df_sample['Total_Trans_Ct'])

df_sample.columns


sns.displot(df_sample, x='Total_Trans_Ct', row='Attrition_Flag')
sns.displot(df_sample, x='Total_Ct_Chng_Q4_Q1', row='Attrition_Flag')
sns.displot(df_sample, x='Total_Revolving_Bal', row='Attrition_Flag')
sns.displot(df_sample, x='Contacts_Count_12_mon', row='Attrition_Flag')
