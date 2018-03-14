# coding=utf-8
"""
处理最原始数据,将其转换为时空数据'data.h5'
"""
import os
import numpy as np
import time
import h5py

DATAPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
BDR = [30.727818, 104.129591, 30.652828, 104.042102]  # 区域边界,顺时针
WIDTH, HEIGHT = 16, 16  # 将区域划分为16*16的子区域
T = 48  # 将一天分为48个时间段,每个时间段30min
TS = []  # 存储全部时间节点
YEAR = 2016
MONTH = 11
DAY = 30
for day in range(1, DAY + 1):
    for t in range(1, T + 1):
        if len(str(day)) == 1:
            day = '0' + str(day)
        if len(str(t)) == 1:
            t = '0' + str(t)
        TS.append(str(YEAR) + str(MONTH) + str(day) + str(t))
# print TS
TEST_DAY = 23
TRAIN_DAY = 7
DATA = np.zeros((T * DAY, 1, 16, 16))


def cut_map():
    udi = (BDR[0] - BDR[2]) / HEIGHT
    lri = (BDR[1] - BDR[3]) / WIDTH
    up2down = []
    left2right = []
    for i in range(HEIGHT):
        up2down.append(BDR[0] - i * udi)
        left2right.append(BDR[3] + i * lri)
    up2down.append(BDR[2])
    left2right.append(BDR[1])
    return up2down, left2right


def load_data():
    for d in range(1, DAY + 1):
        if len(str(d)) == 1:
            f = open(os.path.join(DATAPATH, 'order_2016110' + str(d) + '.txt'))
        else:
            f = open(os.path.join(DATAPATH, 'order_201611' + str(d) + '.txt'))
        lines = f.readlines()
        log = []
        lat = []
        tm = []
        for line in lines:
            driver_id, up_time, down_time, up_log, up_lat, down_log, down_lat = line.split(",")
            tm.append(float(up_time))
            log.append(float(up_log))
            lat.append(float(up_lat))
        lat_interval, log_interval = cut_map()
        row = 0
        col = 0
        for i in range(len(log)):
            if log[i] > 104.129591 or log[i] < 104.042102 or lat[i] > 30.727818 or lat[i] < 30.652828:
                # 若不在区域内
                continue
            time_local = time.localtime(tm[i])
            day = time_local.tm_mday
            hour = time_local.tm_hour
            min = time_local.tm_min
            if min < 30:
                num = (day - 1) * T + hour * T / 24
            else:
                num = (day - 1) * T + hour * T / 24 + 1
            for j in range(1, WIDTH + 1):
                if log[i] < log_interval[j]:
                    col = j - 1
                    break
            # print col
            for j in range(1, HEIGHT + 1):
                if lat[i] > lat_interval[j]:
                    row = j - 1
                    break
            # print row
            DATA[num][0][row][col] += 1
    timestamp = np.asarray(TS)
    data = np.asarray(DATA)
    h5f = h5py.File('../data/data.h5', 'w')
    h5f.create_dataset('timestamp', data=timestamp)
    h5f.create_dataset('data', data=data)
    h5f.close()


load_data()
