import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm




df = pd.read_csv("C://data_minsung/kaggle/bitcoin/bitcoin.csv")



df.Timestamp = pd.to_datetime(df.Timestamp, unit='s')
df.set_index('Timestamp', inplace=True)

df = df.resample('D').mean()

# data describe
df.describe()
# check null value
df[df["Weighted_Price"].isna()==True]

# KNN imputation - impute null data with KNNImputer
Kimp = KNNImputer(n_neighbors=10)
df = pd.DataFrame(Kimp.fit_transform(df), index=df.index, columns=df.columns)

# normalize
def normalize(data):
    mean = data.mean()
    std = data.std()
    return (data-mean)/std

def data_for_rnn(input, seq_len):
    x_li = []
    x_idx = input.index[seq_len-1:-1]
    for i in range(len(input)-seq_len):
        x_li.append(input.iloc[i:i+seq_len].values)
    return x_li, x_idx

def RNNmodel(out_shape, seq_len, n_feature):
    out_shape = out_shape

    model = models.Sequential()
    model.add(layers.Input(shape=(seq_len, n_feature), name='input'))
    model.add(layers.LSTM(128, activation='relu', return_sequences=True, name='first'))
    model.add(layers.LSTM(64, activation='relu', name='second'))
    model.add(layers.Dense(out_shape))
    model.summary()
    model.compile(optimizer='adam',
                  loss = 'mse',
                  metrics=['accuracy'])
    return model



df_norm = normalize(df)
sns.heatmap(df.corr())

# check Multi-Collinearity with VIF
vif = pd.DataFrame()
vif['vif_factor'] = [variance_inflation_factor(df_norm.values, i) for i in range(df_norm.shape[1])]
vif['feature'] = df.columns

# get rid of 'Volume_(Currency)' which has the highest multiCollinearity
df_norm.drop('Volume_(Currency)', axis=1, inplace=True)

cols = df_norm.columns
df_norm.plot()
# check stiotionary with adfuller => every feature is non-stationary feature except 'Volume(BTC)'
for col in cols:
    print("Augmented Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(df_norm[col])[1])
    # sm.tsa.seasonal_decompose(df_norm[col]).plot()


seq_len = 20

x_rnn, x_idx = data_for_rnn(df_norm['Weighted_Price'], seq_len)
x_rnn = tf.convert_to_tensor(np.expand_dims(x_rnn, axis=2))

y = df['Weighted_Price'].iloc[20:]
y = normalize(y)
y_rnn = tf.convert_to_tensor(np.expand_dims(y, axis=1))

trn_size = int(len(x_rnn)*0.7)
val_size = int(len(x_rnn)*0.1)

x_trn = x_rnn[:trn_size]
x_val = x_rnn[trn_size:trn_size+val_size]
x_tst = x_rnn[trn_size+val_size:]


y_trn = y_rnn[:trn_size]
y_val = y_rnn[trn_size:trn_size+val_size]
y_tst = y_rnn[trn_size+val_size:]

model = RNNmodel(out_shape=1, seq_len=seq_len, n_feature=1)
model.summary()
model.fit(x_trn, y_trn, epochs=40, validation_data=(x_val, y_val))
#
# from sklearn.metrics import mean_absolute_error
# mean_absolute_error(y_tst, pred)

pred = model.predict(x_tst)
pred_df = pd.DataFrame(pred, index=y.index[+trn_size+val_size:])
plt.plot(y)
plt.plot(pred_df)
plt.show()
