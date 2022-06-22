import pandas_datareader as pdr
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import math
import quantstats as qs
import numpy as np

# 예측 수익률 (+, -) 유무로 공격자산 방어자산 선택하고
# 각 자산군의 가장 높은 모멘텀 스코어를 뽑아서 투자 진행하기