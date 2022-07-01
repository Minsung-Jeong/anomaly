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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
# import tensorflow as tf
# from tensorflow.keras import datasets, layers, models

os.chdir('C:/data_minsung')

# MAA_data에서 받아온 데이터
# y_cut

t10y2y = pd.read_csv('./finance/new_data/T10Y2Y.csv').set_index('DATE')
t10y3m = pd.read_csv('./finance/new_data/T10Y3M.csv').set_index('DATE')

