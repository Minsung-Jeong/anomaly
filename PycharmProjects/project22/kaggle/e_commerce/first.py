import pandas as pd



df = pd.read_csv("C://data_minsung/kaggle/e_commerce/data.csv",encoding="ISO-8859-1",
                         dtype={'CustomerID': str,'InvoiceID': str})

df.info()
df.describe()
df.head()