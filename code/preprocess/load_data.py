# coding=utf-8
import os
import numpy as np

DATAPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
BDR = [30.727818, 104.129591, 30.652828, 104.042102]  # 顺时针
WIDTH = 16
HEIGHT = 16
T = 48
DATA = np.zeros((16, 16))


def cut_map():
    udi = (BDR[0] - BDR[2])/HEIGHT
    lri = (BDR[1] - BDR[3])/WIDTH
    up2down = []
    left2right = []
    for i in range(HEIGHT):
        up2down.append(BDR[0] - i * udi)
        left2right.append(BDR[3] + i * lri)
    up2down.append(BDR[2])
    left2right.append(BDR[1])
    return up2down, left2right


def load_data():
    f = open(os.path.join(DATAPATH, 'order_20161101.txt'))
    lines = f.readlines()
    log = []
    lat = []
    time = []
    for line in lines:
        driver_id, up_time, down_time, up_log, up_lat, down_log, down_lat = line.split(",")
        time.qppend(up_time)
        log.append(float(up_log))
        lat.append(float(up_lat))
    lat_interval, log_interval = cut_map()
    row = 0
    col = 0
    for i in range(len(log)):
        for j in range(WIDTH):
            if log[i] < log_interval[j]:
                col = j - 1
                break
        # print col
        for j in range(HEIGHT):
            if lat[i] > lat_interval[j]:
                row = j - 1
                break
        # print row
        DATA[row][col] += 1
    print DATA

load_data()
