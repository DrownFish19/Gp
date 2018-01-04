# coding=utf-8
from __future__ import print_function
import os
import h5py
import numpy as np
from ..preprocessing import MinMaxNormalization
from ..datasets.STMatrix import STMatrix

DATAPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..','data')


def load_h5data(fname):
    f = h5py.File(fname, 'r')
    data = f['data'].value
    timestamps = f['timestamp'].value
    f.close()
    return timestamps, data


def load_data(T, len_closeness, len_period, len_test):
    assert (len_closeness + len_period > 0)
    # load data
    fname = os.path.join(DATAPATH, 'test.h5')
    timestamps, data = load_h5data(fname)
    data_all = [data]
    timestamps_all = [timestamps]
    # minmax_scale
    data_train = data[:-len_test]
    print('train_data shape: ', data_train.shape)
    mmn = MinMaxNormalization()
    mmn.fit(data_train)
    data_all_mmn = []
    for d in data_all:
        data_all_mmn.append(mmn.transform(d))

    XC, XP, XT = [], [], []
    Y = []
    timestamps_Y = []
    for data, timestamps in zip(data_all_mmn, timestamps_all):
        # instance-based dataset --> sequences with format as (X, Y) where X is a sequence of images and Y is an image.
        st = STMatrix(data, timestamps, T, CheckComplete=False)
        _XC, _XP, _Y, _timestamps_Y = st.create_dataset(len_closeness=len_closeness, len_period=len_period)
        XC.append(_XC)
        XP.append(_XP)
        # XT.append(_XT)
        Y.append(_Y)
        timestamps_Y += _timestamps_Y

    XC = np.vstack(XC)
    XP = np.vstack(XP)
    # XT = np.vstack(XT)
    Y = np.vstack(Y)
    print("XC shape: ", XC.shape, "XP shape: ", XP.shape,
          # "XT shape: ", XT.shape,
          "Y shape:", Y.shape)

    XC_train, XP_train, Y_train = XC[
                                  :-len_test], XP[:-len_test], Y[:-len_test]
    XC_test, XP_test, Y_test = XC[
                               -len_test:], XP[-len_test:], Y[-len_test:]
    timestamp_train, timestamp_test = timestamps_Y[
                                      :-len_test], timestamps_Y[-len_test:]

    X_train = []
    X_test = []
    for l, X_ in zip([len_closeness, len_period], [XC_train, XP_train]):
        if l > 0:
            X_train.append(X_)
    for l, X_ in zip([len_closeness, len_period], [XC_test, XP_test]):
        if l > 0:
            X_test.append(X_)
    print(len(X_train),len(X_test))
    # print('train shape:', X_train.shape, Y_train.shape,
    #       'test shape: ', X_test.shape, Y_test.shape)
    return X_train, Y_train, X_test, Y_test, mmn, timestamp_train, timestamp_test
