import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C://data_minsung/kaggle/creditcardCustomer/BankChurners.csv")

# ignore it
df = df.iloc[:,:-2]

df.head()
df.info()

df_copy = df.copy()

def convert_num(variable):
    temp = list(set(df_copy[variable]))
    temp_dict = {temp[i]: i for i in range(len(temp))}
    for x in temp_dict:
        df_copy[variable][df_copy[variable] == x] = temp_dict[x]
    return df_copy[variable].astype(float), temp_dict

# 0 = existing Customer , 1 = attrited Customer
df_copy['Attrition_Flag'], attrit_dict = convert_num('Attrition_Flag')
# 0 = female , 1 = male
df_copy['Gender'], gender_dict = convert_num('Gender')
# 0=college, 1=graduate, 2=unknown, 3 = high school, 4= post_gradu, 5=doctor, 6=uneducated
df_copy['Education_Level'], edu_dict = convert_num('Education_Level')
df_copy['Marital_Status'], marital_dict = convert_num('Marital_Status')
df_copy['Income_Category'], income_dict = convert_num('Income_Category')
df_copy['Card_Category'], card_dict = convert_num('Card_Category')


# correlation heatmap
df_copy_corr = df_copy.corr()
plt.figure(figsize=(15,8))
plt.title('heatmap')
sns.heatmap(df_copy_corr, annot=True, cmap="BuPu_r")
plt.show()

"""
# Correlation Top3 variables = Total_trans_Ct, Total_Ct_Chng_Q4_Q1, Total_Revolving_Bal
Total_trans_Ct : total transaction count
Total_Ct_Chng_Q4_Q1 : Change in Transaction Count (Q4 over Q1)
Total_Revolving_Bal : Total Revolving Balance on the Credit Card
Contacts_Count_12_mon : No. of Contacts in the last 12 months
"""
abs(df_copy_corr['Attrition_Flag']).sort_values(ascending=False)
