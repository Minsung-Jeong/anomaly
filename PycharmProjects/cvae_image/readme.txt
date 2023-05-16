cvae 
1차(그대로) : 
loss 177.34
Epoch: 4, Test set ELBO: -175.5509490966797, time elapse for current epoch: 139.49108505249023

2차(dkl 변경)
loss 196.65
Epoch: 4, Test set ELBO: -200.9326171875, time elapse for current epoch: 127.09023928642273


3차 (dkl 아무값)
loss nan


4차 (dkl 상수들을 좀 건드림)
loss -614216787772488456234749394944.00(-inf 을 향해 감)
Epoch: 4, Test set ELBO: inf, time elapse for current epoch: 146.26293206214905
