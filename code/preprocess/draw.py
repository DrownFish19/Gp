# coding=utf-8
import matplotlib.pyplot as plt
import load_data
import os
import numpy as np
import pandas as pd
import seaborn as sns

DATAPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
fname = os.path.join(DATAPATH, 'lastdata.h5')
BDR = [30.727818, 104.129591, 30.652828, 104.042102]  # 顺时针
WIDTH = 16
HEIGHT = 16
T = 48
TS = []
YEAR = 2016
MONTH = 11
DAY = 30
WK = ['Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'Mon']

# print (BDR[1]+BDR[3])/2,(BDR[0]+BDR[2])/2
ts, data = load_data.load_h5data(fname)
# # heatmap_all
# data_all = np.zeros((16, 16))
# for row in data:
#     data_all += row
# data_all = pd.DataFrame(data_all)
# print data_all
# f, ax = plt.subplots(figsize = (16, 16))
# sns.heatmap(data_all,vmax=50000,cmap=plt.cm.Reds)
# f.savefig('../pic/heatmap_all.png', bbox_inches='tight')

# # heatmap_weekdayandweekend
# data_wk2 = np.zeros((16, 16))
# for row in data[T*2:T*3,:,:]:
#     data_wk2 += row
# data_wkd5 = np.zeros((16, 16))
# for row in data[T*5:T*6,:,:]:
#     data_wkd5 += row
# f, ax = plt.subplots(figsize = (16, 16))
# sns.heatmap(data_wk2,vmax=1600,cmap=plt.cm.Reds)
# f.savefig('../pic/heatmap_wk2.png', bbox_inches='tight')
# f, ax = plt.subplots(figsize = (16, 16))
# sns.heatmap(data_wkd5,vmax=1600,cmap=plt.cm.Reds)
# f.savefig('../pic/heatmap_wkd5.png', bbox_inches='tight')
# f, ax = plt.subplots(figsize = (16, 16))
# sns.heatmap(data_wkd5-data_wk2,cmap=plt.cm.jet)
# f.savefig('../pic/heatmap_wk_sub.png', bbox_inches='tight')

# data_wkhigh = np.zeros(30)
# data_wklow = np.zeros(30)
# for i in range(30):
#     for a in data[48 * i:48 * (i + 1), 12, 5].flat:
#         data_wkhigh[i] += a
# print data_wkhigh
# for i in range(30):
#     for a in data[48 * i:48 * (i + 1), 12, 12].flat:
#         data_wklow[i] += a
# print data_wklow
# f, ax = plt.subplots()
# data_wkhigh = pd.DataFrame(data_wkhigh,range(1,31))
# plt.plot(data_wkhigh)
# f.savefig('../pic/line_wkhigh.png', bbox_inches='tight')
# f, ax = plt.subplots()
# data_wklow = pd.DataFrame(data_wklow,range(1,31))
# plt.plot(data_wklow)
# f.savefig('../pic/line_wkdhigh.png', bbox_inches='tight')

# data_line = np.zeros(30)
# for i in range(30):
#     for a in data[48*i:48*(i+1),:,:].flat:
#         data_line[i] += a
# print data_line
# f, ax = plt.subplots()
# plt.plot(data_line)
# f.savefig('../pic/line_all.png', bbox_inches='tight')


# data_line = np.zeros(7*48)
# for i in range(7*48):
#     for a in data[i,12,12].flat:
#         data_line[i] += a
# print data_line
# f, ax = plt.subplots()
# plt.xticks(np.asarray(range(7))*48+24,WK)
# plt.plot(data_line)
# f.savefig('../pic/line_wkdhbystamp.png', bbox_inches='tight')

data_line = np.zeros(30)
for i in range(30):
    for a in data[30 * i, 12, 5].flat:
        data_line[i] += a
print data_line
f, ax = plt.subplots()
# plt.xticks(range(0, 48, 2), range(0, 24))
plt.plot(data_line)
f.savefig('../pic/dot_trend125.png', bbox_inches='tight')
