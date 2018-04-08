# coding=utf-8
from __future__ import print_function

import os

from mydeepst.datasets.load_data import load_data_cp, load_data_c, load_data_cpm
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from mydeepst.models.STResNet import stresnet, stresnet_c, stresnet_cpm
import mydeepst.metrics as metrics
import tensorflow as tf

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

T = 48
height, width = 16, 16
len_closeness = 3  # length of closeness dependent sequence
len_period = 4  # length of period dependent sequence
# len_trend = 4  # length of trend dependent sequence
nb_residual_unit = 4  # number of residual units
lr = 0.0002  # learning rate
nb_epoch = 500  # number of epoch at training stage
nb_epoch_cont = 100
batch_size = 16  # batch size
days_test = 7
len_test = T * days_test


def build_model_cp():
    c_conf = (len_closeness, height,
              width) if len_closeness > 0 else None
    p_conf = (len_period, height,
              width) if len_period > 0 else None
    # t_conf = (len_trend,  height,
    #           width) if len_trend > 0 else None

    model = stresnet(c_conf=c_conf, p_conf=p_conf, nb_residual_unit=nb_residual_unit)
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])
    model.summary()
    return model


def build_model_cpm(external_dim):
    c_conf = (len_closeness, height,
              width) if len_closeness > 0 else None
    p_conf = (len_period, height,
              width) if len_period > 0 else None
    # t_conf = (len_trend,  height,
    #           width) if len_trend > 0 else None

    model = stresnet_cpm(c_conf=c_conf, p_conf=p_conf, external_dim=external_dim, nb_residual_unit=nb_residual_unit)
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])
    model.summary()
    return model


def build_model_c():
    c_conf = (len_closeness, height,
              width) if len_closeness > 0 else None
    # p_conf = (len_period, height,
    #           width) if len_period > 0 else None
    # t_conf = (len_trend,  height,
    #           width) if len_trend > 0 else None

    model = stresnet_c(c_conf=c_conf, nb_residual_unit=nb_residual_unit)
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])
    model.summary()
    return model


def deepST_CP():
    # load data
    print("loading data...")
    X_train, Y_train, X_test, Y_test, mmn, timestamp_train, timestamp_test = load_data_cp(
        T=T, len_closeness=len_closeness, len_period=len_period, len_test=len_test)

    print('=' * 10)
    print("compiling model...")
    model = build_model_cp()
    hyperparams_name = 'c{}.p{}.resunit{}.lr{}'.format(
        len_closeness, len_period, nb_residual_unit, lr)
    fname_param = os.path.join('result', '{}.best.h5'.format(hyperparams_name))
    early_stopping = EarlyStopping(monitor='val_rmse', patience=5, mode='min')
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')

    print('=' * 10)
    print("training model...")
    history = model.fit(X_train, Y_train,
                        nb_epoch=nb_epoch,
                        batch_size=batch_size,
                        validation_split=0.1,
                        callbacks=[early_stopping, model_checkpoint],
                        verbose=1)
    model.save_weights(os.path.join(
        'result', '{}.h5'.format(hyperparams_name)), overwrite=True)

    print('=' * 10)
    print('evaluating using the model that has the best loss on the valid set')
    score = model.evaluate(X_train, Y_train, batch_size=Y_train.shape[
                                                            0] // 48, verbose=0)
    print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
    score = model.evaluate(
        X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
    print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))

    print('=' * 10)
    print("training model (cont)...")
    fname_param = os.path.join(
        'result', '{}.cont.best.h5'.format(hyperparams_name))
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor='rmse', verbose=0, save_best_only=True, mode='min')
    history = model.fit(X_train, Y_train, nb_epoch=nb_epoch_cont, verbose=1, batch_size=batch_size, callbacks=[
        model_checkpoint], validation_data=(X_test, Y_test))

    model.save_weights(os.path.join(
        'result', '{}_cont.h5'.format(hyperparams_name)), overwrite=True)

    print('=' * 10)
    print('evaluating using the final model')
    score = model.evaluate(
        X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
    print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))


def deepST_CPM():
    # load data
    print("loading data...")
    X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = load_data_cpm(
        T=T, len_closeness=len_closeness, len_period=len_period, len_test=len_test,
        meta_data=True, meteorol_data=True, holiday_data=False)

    print('=' * 10)
    print("compiling model...")
    model = build_model_cpm(external_dim)
    hyperparams_name = 'c{}.p{}.resunit{}.lr{}.M'.format(
        len_closeness, len_period, nb_residual_unit, lr)
    fname_param = os.path.join('result', '{}.best.h5'.format(hyperparams_name))
    early_stopping = EarlyStopping(monitor='val_rmse', patience=5, mode='min')
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')

    print('=' * 10)
    print("training model...")
    history = model.fit(X_train, Y_train,
                        nb_epoch=nb_epoch,
                        batch_size=batch_size,
                        validation_split=0.1,
                        callbacks=[early_stopping, model_checkpoint],
                        verbose=1)
    model.save_weights(os.path.join(
        'result', '{}.h5'.format(hyperparams_name)), overwrite=True)

    print('=' * 10)
    print('evaluating using the model that has the best loss on the valid set')
    score = model.evaluate(X_train, Y_train, batch_size=Y_train.shape[
                                                            0] // 48, verbose=0)
    print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
    score = model.evaluate(
        X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
    print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))

    print('=' * 10)
    print("training model (cont)...")
    fname_param = os.path.join(
        'result', '{}.cont.best.h5'.format(hyperparams_name))
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor='rmse', verbose=0, save_best_only=True, mode='min')
    history = model.fit(X_train, Y_train, nb_epoch=nb_epoch_cont, verbose=1, batch_size=batch_size, callbacks=[
        model_checkpoint], validation_data=(X_test, Y_test))

    model.save_weights(os.path.join(
        'result', '{}_cont.h5'.format(hyperparams_name)), overwrite=True)

    print('=' * 10)
    print('evaluating using the final model')
    score = model.evaluate(
        X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
    print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))


def deepST_C():
    # load data
    print("loading data...")
    # X_train, Y_train, X_test, Y_test, mmn, timestamp_train, timestamp_test = load_data(
    #     T=T, len_closeness=len_closeness, len_period=len_period, len_test=len_test)
    X_train, Y_train, X_test, Y_test, mmn, timestamp_train, timestamp_test = load_data_c(
        T=T, len_closeness=len_closeness, len_test=len_test)

    print('=' * 10)
    print("compiling model...")
    model = build_model_c()
    hyperparams_name = 'c{}.resunit{}.lr{}'.format(
        len_closeness, nb_residual_unit, lr)
    fname_param = os.path.join('result', '{}.best.h5'.format(hyperparams_name))
    early_stopping = EarlyStopping(monitor='val_rmse', patience=5, mode='min')
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')

    print('=' * 10)
    print("training model...")
    history = model.fit(X_train, Y_train,
                        nb_epoch=nb_epoch,
                        batch_size=batch_size,
                        validation_split=0.1,
                        callbacks=[early_stopping, model_checkpoint],
                        verbose=1)
    model.save_weights(os.path.join(
        'result', '{}.h5'.format(hyperparams_name)), overwrite=True)

    print('=' * 10)
    print('evaluating using the model that has the best loss on the valid set')
    score = model.evaluate(X_train, Y_train, batch_size=Y_train.shape[
                                                            0] // 48, verbose=0)
    print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
    score = model.evaluate(
        X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
    print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))

    print('=' * 10)
    print("training model (cont)...")
    fname_param = os.path.join(
        'result', '{}.cont.best.h5'.format(hyperparams_name))
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor='rmse', verbose=0, save_best_only=True, mode='min')
    history = model.fit(X_train, Y_train, nb_epoch=nb_epoch_cont, verbose=1, batch_size=batch_size, callbacks=[
        model_checkpoint], validation_data=(X_test, Y_test))

    model.save_weights(os.path.join(
        'result', '{}_cont.h5'.format(hyperparams_name)), overwrite=True)

    print('=' * 10)
    print('evaluating using the final model')
    score = model.evaluate(
        X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
    print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))


if __name__ == '__main__':
    # deepST_C()
    # deepST_CP()
    deepST_CPM()
