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
# 모양 변형 전에 월별데이터로 만들고 잘라야 함 - 
aa = pd.DataFrame(x_test, index=idx_test)
aa.resample(rule='M').last()