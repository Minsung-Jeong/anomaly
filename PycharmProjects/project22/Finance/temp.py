import pandas as pd
import numpy as np
x_cut
y_cut

x_rnn
y_rnn
np.shape(x_rnn)
len(x_cut)
len(x_rnn)

idx_x

len(x_rnn)
len(idx_rnn)

x_test = x_rnn[train_size:]
idx_test = pd.to_datetime(idx_rnn)[train_size:]


pd.to_datetime(idx_rnn)[0]

# ValueError: Must pass 2-d input. shape=(492, 30, 14) - 3-d 데이터는 pd 못 넣음
# 변형 전에 학습-테스트 나누고, 테스트는 월별데이터로 만든 다음 변형
trn_prop = 0.8
train_size = round(len(x_cut)*trn_prop)
x_cut = x_cut

idx_train = pd.to_datetime(idx_x).values[:train_size]
idx_test = pd.to_datetime(idx_x).values[train_size:]

x_train = x_cut[:train_size].values
x_test = x_cut[train_size:].values

x_pd = pd.DataFrame(x_test, index=idx_test)
x_m = x_pd.resample(rule='M').last()
len(x_pd)
# y값
y_train = y_cut[:train_size]
y_test = y_cut[train_size:]
y_m = y_test.resample(rule='M').last()

# 테스트 결과 다루기
# 5, -10은 데이터 월 단위로 이쁘게 잘라보려고
new_idx = tst_index[5:-10]
rnn_bin = get_binary(RNN_test)[5:-10]
decision = pd.DataFrame(rnn_bin, index=tst_index).resample(rule='M').mean()

# 월별로 해서 0 / 1중 더 많은 관측치가 나온 것을 그 달의 선택으로
set(new_idx.year)
new_idx.month


