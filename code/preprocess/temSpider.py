# coding=utf-8
"""
爬虫,从www.wunderground.com网站爬取历史天气信息,包括气温/风速/天气
结果保存为'ex.h5'
"""
import sys
import urllib2
from bs4 import BeautifulSoup
import h5py
import numpy as np

reload(sys)
sys.setdefaultencoding('utf8')

ex = []
WR_ALL = ['Clear','Scattered Clouds','Mist','Light Rain','Partly Cloudy','Fog','Unknown','Haze'
,'Mostly Cloudy','Light Rain Showers','Patches of Fog']

def do_spider(url, count):
    try:
        req = urllib2.Request(url)
        source_code = urllib2.urlopen(req).read()
        plain_text = str(source_code)
    except (urllib2.HTTPError, urllib2.URLError), e:
        print e

    soup = BeautifulSoup(plain_text, "html5lib")
    list_soup = soup.find('table', {'class': 'obs-table'})

    for tr_info in list_soup.findAll('tr')[1:]:
        print count
        td_info = tr_info.findAll('td')
        qiwen = td_info[1].find('span', {'class': 'wx-value'}).string.strip().encode()
        fengsu = td_info[7]
        if fengsu.findAll('span', {'class': 'wx-value'}):
            fengsu = fengsu.findAll('span', {'class': 'wx-value'})[1].string.strip().encode()
        else:
            fengsu = '0.0'

        tianqi = td_info[-1].string.strip().encode()
        ex.append([qiwen, fengsu, tianqi])
        ex.append([qiwen, fengsu, tianqi])


if __name__ == '__main__':
    for i in range(1, 31):
        print i
        do_spider(
            url='https://www.wunderground.com/history/airport/ZUUU/2016/11/' + str(
                i) + '/DailyHistory.html?req_city=Chengdu&req_statename=China',
            count=i)
    f = h5py.File('../data/ex.h5', 'w')
    f.create_dataset('ex', data=ex)
    ex = f['ex'].value
    WS = []
    TE = []
    WR = []
    for i in ex:
        WS.append(float(i[0]))
        TE.append(float(i[1]))
        temp = np.zeros(11)
        for j in range(11):
            if i[2] == WR_ALL[j]:
                temp[j] = 1
                break
        WR.append(temp)
    WS = np.asarray(WS)
    WR = np.asarray(WR)
    TE = np.asarray(TE)
    h5f = h5py.File('../data/lastex.h5', 'w')
    h5f.create_dataset('WS', data=WS)
    h5f.create_dataset('TE', data=TE)
    h5f.create_dataset('WR', data=WR)
    h5f.close()
