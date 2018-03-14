# coding=utf-8
from __future__ import print_function
import os
import h5py
import numpy as np
from ..preprocessing import MinMaxNormalization, timestamp2vec
from ..datasets.STMatrix import STMatrix

DATAPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')


def load_h5data(fname):
    f = h5py.File(fname, 'r')
    data = f['data'].value
    timestamps = f['timestamp'].value
    f.close()
    return timestamps, data


def load_meteorol(timeslots, fname=os.path.join(DATAPATH, 'lastex.h5')):
    '''
    timeslots: the predicted timeslots
    In real-world, we dont have the meteorol data in the predicted timeslot, instead, we use the meteoral at previous timeslots, i.e., slot = predicted_slot - timeslot (you can use predicted meteorol data as well)
    '''
    f = h5py.File(fname, 'r')
    WindSpeed = f['WS'].value
    Weather = f['WR'].value
    Temperature = f['TE'].value
    f.close()

    # M = dict()  # map timeslot to index
    # for i, slot in enumerate(Timeslot):
    #     M[slot] = i

    # WS = []  # WindSpeed
    # WR = []  # Weather
    # TE = []  # Temperature
    # for slot in timeslots:
    #     predicted_id = M[slot]
    #     cur_id = predicted_id - 1
    #     WS.append(WindSpeed[cur_id])
    #     WR.append(Weather[cur_id])
    #     TE.append(Temperature[cur_id])

    WS = np.asarray(WindSpeed)
    WR = np.asarray(Weather)
    TE = np.asarray(Temperature)

    # 0-1 scale
    WS = 1. * (WS - WS.min()) / (WS.max() - WS.min())
    TE = 1. * (TE - TE.min()) / (TE.max() - TE.min())

    print("shape: ", WS.shape, WR.shape, TE.shape)

    # concatenate all these attributes
    merge_data = np.hstack([WR, WS[:, None], TE[:, None]])

    print('meger shape:', merge_data.shape)
    return merge_data[-len(timeslots):]


def load_data_cp(T, len_closeness, len_period, len_test):
    assert (len_closeness + len_period > 0)
    # load data
    fname = os.path.join(DATAPATH, 'data.h5')
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
    print(len(X_train), len(X_test))
    # print('train shape:', X_train.shape, Y_train.shape,
    #       'test shape: ', X_test.shape, Y_test.shape)
    return X_train, Y_train, X_test, Y_test, mmn, timestamp_train, timestamp_test


def load_data_cpm(T, len_closeness, len_period, len_test, meta_data=True, meteorol_data=True, holiday_data=False):
    assert (len_closeness + len_period > 0)
    # load data
    fname = os.path.join(DATAPATH, 'data.h5')
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
    meta_feature = []
    print(timestamps_Y)
    if meta_data:
        # load time feature
        time_feature = timestamp2vec(timestamps_Y)
        meta_feature.append(time_feature)
    if meteorol_data:
        # load meteorol data
        meteorol_feature = load_meteorol(timestamps_Y)
        meta_feature.append(meteorol_feature)
    meta_feature = np.hstack(meta_feature) if len(
        meta_feature) > 0 else np.asarray(meta_feature)
    metadata_dim = meta_feature.shape[1] if len(
        meta_feature.shape) > 1 else None
    if metadata_dim < 1:
        metadata_dim = None
    if meta_data and holiday_data and meteorol_data:
        print('time feature:', time_feature.shape,
              'meteorol feature: ', meteorol_feature.shape, 'meta feature: ', meta_feature.shape)

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

    if metadata_dim is not None:
        meta_feature_train, meta_feature_test = meta_feature[
                                                :-len_test], meta_feature[-len_test:]
        X_train.append(meta_feature_train)
        X_test.append(meta_feature_test)
    print(len(X_train), len(X_test))
    # print('train shape:', X_train.shape, Y_train.shape,
    #       'test shape: ', X_test.shape, Y_test.shape)
    return X_train, Y_train, X_test, Y_test, mmn, metadata_dim, timestamp_train, timestamp_test


def load_data_c(T, len_closeness, len_test):
    assert (len_closeness > 0)
    # load data
    fname = os.path.join(DATAPATH, 'data.h5')
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
        _XC, _Y, _timestamps_Y = st.create_dataset_c(len_closeness=len_closeness)
        XC.append(_XC)
        # XP.append(_XP)
        # XT.append(_XT)
        Y.append(_Y)
        timestamps_Y += _timestamps_Y

    XC = np.vstack(XC)
    # XP = np.vstack(XP)
    # XT = np.vstack(XT)
    Y = np.vstack(Y)
    print("XC shape: ", XC.shape,
          # "XP shape: ", XP.shape,
          # "XT shape: ", XT.shape,
          "Y shape:", Y.shape)

    XC_train, Y_train = XC[
                        :-len_test], Y[:-len_test]
    XC_test, Y_test = XC[
                      -len_test:], Y[-len_test:]
    timestamp_train, timestamp_test = timestamps_Y[
                                      :-len_test], timestamps_Y[-len_test:]

    X_train = []
    X_test = []
    for l, X_ in zip([len_closeness], [XC_train]):
        if l > 0:
            X_train.append(X_)
    for l, X_ in zip([len_closeness], [XC_test]):
        if l > 0:
            X_test.append(X_)
    print(len(X_train), len(X_test))
    # print('train shape:', X_train.shape, Y_train.shape,
    #       'test shape: ', X_test.shape, Y_test.shape)
    return X_train, Y_train, X_test, Y_test, mmn, timestamp_train, timestamp_test
