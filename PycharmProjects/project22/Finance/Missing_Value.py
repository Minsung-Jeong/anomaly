import pandas_datareader as pdr
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import math
import quantstats as qs
import numpy as np
from sklearn import preprocessing
import os

start_date = '2000-01-01'
end_date = '2022-04-15'

os.chdir('C:/data_minsung')



t10y2y = pd.read_csv('./finance/new_data/T10Y2Y.csv').set_index('DATE')
t10y3m = pd.read_csv('./finance/new_data/T10Y3M.csv').set_index('DATE')
acogno = pd.read_csv('./finance/new_data/ACOGNO.csv').set_index('DATE')

