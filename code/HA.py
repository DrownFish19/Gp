# coding=utf-8
import numpy as np
from mydeepst.datasets.load_data import timeseries

T = 48
days_test = 7
days_train = 23
len_test = T * days_test

ts, data = timeseries()
test_data = data[:, -len_test:]
test_timestamps = ts[-len_test:]

avg_data = np.zeros((data.shape[0], len_test))
sum = 0
i = 1
for n in range(test_data.shape[0]):
    for t in range(test_data.shape[1]):
        while i:
            sum += data[n, days_train * T + t - i * T]
            if (days_train * T + t - (i + 1) * T < 0):
                break
            i = i + 1
        avg_data[n, t] = sum / i
        sum = 0
        i = 1
n_rmse_val = np.sqrt(np.sum(np.square(test_data-avg_data))*1.0/np.prod(test_data.shape))
print('The RMSE of HA on database BikeNYC is :',n_rmse_val)
