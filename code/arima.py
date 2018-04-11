# coding=utf-8
from __future__ import print_function
import numpy as np
import pandas as pd
from time import mktime
import time
from datetime import datetime
import pandas as pd
from mydeepst.preprocessing.minmax_normalization import MinMaxNormalization
from mydeepst.datasets.load_data import timeseries
from statsmodels.tsa.arima_model import ARMA


T = 48
days_test = 7
days_train = 23
len_test = T * days_test

input_steps = 10
output_steps = 1
run_times = 500
pre_process = MinMaxNormalization()
split = [T*days_train,T*days_test]
print('load train, validate, test data...')
ts, data = timeseries()
print('preprocess train data...')
pre_process.fit(data)  # 返回最大值和最小值
data = pre_process.transform(data)
# data = np.transpose(np.asarray(data), (1,0))
#(256,48*30)
all_timestamps_struct = [time.strptime(t, '%Y%m%d%H%M') for t in ts]
timestamps = [datetime.fromtimestamp(mktime(t)) for t in all_timestamps_struct]
train_data = data[ :-len_test,:]
test_data = data[-len_test:,:]
train_timestamps = timestamps[:-len_test]
test_timestamps = timestamps[-len_test:]


print('======================= ARMA for test ===============================')
loss = 0
error_count = 0
index_all = np.zeros([run_times, 2])
error_index = np.zeros(run_times)
test_target = np.zeros([run_times, output_steps])
test_prediction = np.zeros([run_times, output_steps])
for r in range(run_times):
    print('run '+str(r))
    i = np.random.randint(data.shape[0])
    j = np.random.randint(test_data.shape[-1]-output_steps)
    train_df = pd.DataFrame(data[i][j:split[0]+j])
    train_df.index = pd.DatetimeIndex(timestamps[j:split[0]+j])

    try:
        results = ARMA(train_df, order=(2,2)).fit(trend='nc', disp=-1)
    except:
        error_index[error_count] = r
        error_count += 1
        continue
    pre, _, _ = results.forecast(output_steps)
    test_real = test_data[i][j:j+output_steps]
    index_all[r] = [i,j]
    test_target[r] = test_real
    test_prediction[r] = pre
    loss += np.sum(np.square(pre - test_real))
print('================ calculate rmse for test data ============')
#n_rmse_val = np.sqrt(np.sum(np.square(val_predict - val_real))*1.0/np.prod(val_real.shape))
#n_rmse_test = np.sqrt(np.sum(np.square(test_predict - test_real))*1.0/np.prod(test_real.shape))
#rmse_val = pre_process.real_loss(n_rmse_val)
#rmse_test = pre_process.real_loss(n_rmse_test)
#print('val loss is ' + str(n_rmse_val) + ' , ' + str(rmse_val))
#print('test loss is ' + str(n_rmse_test) + ' , ' + str(rmse_test))
#print('val loss is ' + str(n_rmse_val))
print('run times: '+str(run_times))
print('error count: '+str(error_count))
rmse = np.sqrt(loss/((run_times-error_count)*output_steps))
rmse = rmse* (pre_process._max - pre_process._min) / 2
print('test loss is ' + str(rmse))
# np.save('../citybike-results/results/ARMA/test_target.npy', test_target)
# np.save('../citybike-results/results/ARMA/test_prediction.npy', test_prediction)
# np.save('../citybike-results/results/ARMA/index_all.npy', index_all)
# np.save('../citybike-results/results/ARMA/error_index.npy', error_index)