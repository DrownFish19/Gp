# coding=utf-8
from __future__ import print_function
import numpy as np
import pandas as pd
from scipy import stats
from time import mktime
from datetime import datetime
import tensorflow as tf
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
import sys
from mydeepst.datasets.load_data import timeseries
from mydeepst.preprocessing import *
from mydeepst.utils import *
import h5py

T = 48
days_test = 7
days_train = 23
len_test = T * days_test

input_steps = 10
output_steps = 1
pre_process = MinMaxNormalization()

print('load train, validate, test data...')
ts, data = timeseries()
print('preprocess train data...')
pre_process.fit(data)  # 返回最大值和最小值
data = pre_process.transform(data)
data = np.transpose(np.asarray(data), (1,0))
all_timestamps_struct = [time.strptime(t, '%Y%m%d%H%M') for t in ts]
timestamps = [datetime.fromtimestamp(mktime(t)) for t in all_timestamps_struct]
train_data = data[ :-len_test,:]
test_data = data[-len_test:,:]
train_timestamps = timestamps[:-len_test]
test_timestamps = timestamps[-len_test:]


column_name = [str(e) for e in range(1, train_data.shape[1] + 1)]
train_df = pd.DataFrame(train_data, columns=column_name)
train_df.index = pd.DatetimeIndex(train_timestamps)

print(train_df)
print('create VAR model and fit...')
model_var = VAR(train_df)
results = model_var.fit(1)

print('test trained VAR model...')
# lag_order = 1
lag_order = results.k_ar
# val_data_preindex.shape = (1247,256),train_data[-lag_order:]= (1,256)
# val_data_preindex = np.vstack((train_data[-lag_order:], val_data))
# test_data_preindex = (241,256)
test_data_preindex = np.vstack((train_data[-lag_order:], test_data))
# validate and test data
val_real = []
val_predict = []
test_real = []
test_predict = []
for i in range(test_data.shape[0] - output_steps):
    val_predict.append(results.forecast(test_data_preindex[i:i + lag_order], output_steps))
    val_real.append(test_data[i: i + output_steps])
for i in range(test_data.shape[0] - output_steps):
    test_real.append(test_data[i: i + output_steps])
    test_predict.append(results.forecast(test_data_preindex[i:i + lag_order], output_steps))
# 将四个list转化成array
val_real = np.array(val_real)
val_predict = np.array(val_predict)
test_real = np.array(test_real)
test_predict = np.array(test_predict)

# n_rmse_val = np.sqrt(np.sum(np.square(val_predict - val_real)) * 1.0 / np.prod(val_real.shape))
n_rmse_test = np.sqrt(np.sum(np.square(test_predict - test_real)) * 1.0 / np.prod(test_real.shape))
n_rmse_test = n_rmse_test * (pre_process._max - pre_process._min) / 2
# rmse_val = pre_process.real_loss(n_rmse_val)
# rmse_test = pre_process.real_loss(n_rmse_test)
# print('val loss is ' + str(n_rmse_val) + ' , ' + str(rmse_val))
print('test loss is ' + str(n_rmse_test) )
# np.save('../citybike-results/results/VAR/test_target.npy', test_real)
# np.save('../citybike-results/results/VAR/test_prediction.npy', test_predict)
