import pandas as pd

df = pd.read_csv("C://data_minsung/kaggle/creditcardCustomer/BankChurners.csv")

# ignore it
df = df.iloc[:,:-2]
df.columns