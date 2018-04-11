# coding=utf-8
from __future__ import print_function
import os

from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, Activation, Flatten, Concatenate, Reshape, LSTM, Dense

import attention
from mydeepst.datasets.load_data import load_h5data
from mydeepst.preprocessing.minmax_normalization import MinMaxNormalization
import numpy as np
import tensorflow as tf

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

T = 48
days_test = 7
len_test = T * days_test


# class CustomStopper(EarlyStopping):
#     # add argument for starting epoch
#     def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', start_epoch=40):
#         super().__init__(monitor=monitor, min_delta=min_delta, patience=patience, verbose=verbose, mode=mode)
#         self.start_epoch = start_epoch
#
#     def on_epoch_end(self, epoch, logs=None):
#         if epoch > self.start_epoch:
#             super().on_epoch_end(epoch, logs)


def eval_lstm(y, pred_y):
    pickup_y = y[:, 0]
    pickup_pred_y = pred_y[:, 0]
    # pickup_mask = pickup_y > threshold
    # pickup part
    if np.sum(pickup_y) != 0:
        # avg_pickup_mape = np.mean(np.abs(pickup_y[pickup_y]-pickup_pred_y[pickup_y])/pickup_y[pickup_y])
        avg_pickup_rmse = np.sqrt(np.mean(np.square(pickup_y - pickup_pred_y)))

    return avg_pickup_rmse


class file_loader:
    def __init__(self):
        self.isVolumeLoaded = False
        self.isFlowLoaded = False

    def load_volume(self):
        # shape (timeslot_num, type=1, x_num, y_num)
        self.ts, self.dt = load_h5data('data.h5')
        self.ts_train, self.volume_train = self.ts[:-len_test], self.dt[:-len_test]
        self.ts_test, self.volume_test = self.ts[-len_test:], self.dt[-len_test:]

    def sample_Att_ConvLSTM(self, datatype, att_lstm_num=3, long_term_lstm_seq_len=3, short_term_lstm_seq_len=7, \
                            hist_feature_daynum=7, last_feature_num=48, nbhd_size=1, cnn_nbhd_size=3):
        if self.isVolumeLoaded is False:
            self.load_volume()

        if datatype == "train":
            data = self.dt
            time_start = (hist_feature_daynum + att_lstm_num) * T + long_term_lstm_seq_len
            time_end = data.shape[0]
            # time_end=500
        elif datatype == "test":
            data = self.dt
            time_end = data.shape[0]
            time_start = time_end-len_test
        else:
            print("Please select **train** or **test**")
            raise Exception

        cnn_att_features = []
        lstm_att_features = []
        for i in range(att_lstm_num):
            lstm_att_features.append([])
            cnn_att_features.append([])
            for j in range(long_term_lstm_seq_len):
                cnn_att_features[i].append([])

        cnn_features = []
        for i in range(short_term_lstm_seq_len):
            cnn_features.append([])
        short_term_lstm_features = []
        labels = []

        # time_start = (hist_feature_daynum + att_lstm_num) * T + long_term_lstm_seq_len
        # time_end = data.shape[0]
        # time_end=500
        volume_type = data.shape[1]
        mmn = MinMaxNormalization()
        print(time_start, time_end)
        for t in range(time_start, time_end):
            if t % 100 == 0:
                print("Now sampling at {0} timeslots.".format(t))
            for x in range(data.shape[2]):
                for y in range(data.shape[3]):
                    # sample common (short-term) lstm
                    short_term_lstm_samples = []
                    for seqn in range(short_term_lstm_seq_len):
                        # real_t from (t - short_term_lstm_seq_len) to (t-1)
                        real_t = t - (short_term_lstm_seq_len - seqn)

                        # cnn features, zero_padding
                        cnn_feature = np.zeros((volume_type, 2 * cnn_nbhd_size + 1, 2 * cnn_nbhd_size + 1))
                        # actual idx in data
                        for cnn_nbhd_x in range(x - cnn_nbhd_size, x + cnn_nbhd_size + 1):
                            for cnn_nbhd_y in range(y - cnn_nbhd_size, y + cnn_nbhd_size + 1):
                                # boundary check
                                if not (0 <= cnn_nbhd_x < data.shape[2] and 0 <= cnn_nbhd_y < data.shape[3]):
                                    continue
                                # get features
                                cnn_feature[:, cnn_nbhd_x - (x - cnn_nbhd_size), cnn_nbhd_y - (y - cnn_nbhd_size)
                                ] = data[real_t, :, cnn_nbhd_x, cnn_nbhd_y]
                        cnn_features[seqn].append(cnn_feature)

                        # lstm features
                        # nbhd feature, zero_padding
                        nbhd_feature = np.zeros((volume_type, 2 * nbhd_size + 1, 2 * nbhd_size + 1))
                        # actual idx in data
                        for nbhd_x in range(x - nbhd_size, x + nbhd_size + 1):
                            for nbhd_y in range(y - nbhd_size, y + nbhd_size + 1):
                                # boundary check
                                if not (0 <= nbhd_x < data.shape[2] and 0 <= nbhd_y < data.shape[3]):
                                    continue
                                # get features
                                nbhd_feature[:, nbhd_x - (x - nbhd_size), nbhd_y - (y - nbhd_size)] = data[real_t, :,
                                                                                                      nbhd_x, nbhd_y]
                        nbhd_feature = nbhd_feature.flatten()

                        # last feature
                        last_feature = data[real_t - last_feature_num: real_t, :, x, y].flatten()

                        # hist feature
                        hist_feature = data[real_t - hist_feature_daynum * T: real_t: T, :, x, y].flatten()

                        feature_vec = np.concatenate((hist_feature, last_feature))
                        feature_vec = np.concatenate((feature_vec, nbhd_feature))

                        short_term_lstm_samples.append(feature_vec)
                    short_term_lstm_features.append(np.array(short_term_lstm_samples))

                    # sample att-lstms
                    for att_lstm_cnt in range(att_lstm_num):#3

                        # sample lstm at att loc att_lstm_cnt
                        long_term_lstm_samples = []
                        # get time att_t, move forward for (att_lstm_num - att_lstm_cnt) day, then move back for ([long_term_lstm_seq_len / 2] + 1)
                        # notice that att_t-th timeslot will not be sampled in lstm
                        # e.g., **** (att_t - 3) **** (att_t - 2) (yesterday's t) **** (att_t - 1) **** (att_t) (this one will not be sampled)
                        # sample att-lstm with seq_len = 3
                        att_t = t - (att_lstm_num - att_lstm_cnt) * T + (long_term_lstm_seq_len - 1) / 2 + 1
                        att_t = int(att_t)
                        # att-lstm seq len
                        for seqn in range(long_term_lstm_seq_len):
                            # real_t from (att_t - long_term_lstm_seq_len) to (att_t - 1)
                            real_t = att_t - (long_term_lstm_seq_len - seqn)

                            # cnn features, zero_padding
                            cnn_feature = np.zeros((volume_type, 2 * cnn_nbhd_size + 1, 2 * cnn_nbhd_size + 1))
                            # actual idx in data
                            for cnn_nbhd_x in range(x - cnn_nbhd_size, x + cnn_nbhd_size + 1):
                                for cnn_nbhd_y in range(y - cnn_nbhd_size, y + cnn_nbhd_size + 1):
                                    # boundary check
                                    if not (0 <= cnn_nbhd_x < data.shape[2] and 0 <= cnn_nbhd_y < data.shape[3]):
                                        continue
                                    # get features
                                    # import ipdb; ipdb.set_trace()
                                    cnn_feature[:, cnn_nbhd_x - (x - cnn_nbhd_size), cnn_nbhd_y - (y - cnn_nbhd_size)
                                    ] = data[real_t, :, cnn_nbhd_x, cnn_nbhd_y]

                            cnn_att_features[att_lstm_cnt][seqn].append(cnn_feature)

                            # att-lstm features
                            # nbhd feature, zero_padding
                            nbhd_feature = np.zeros((volume_type, 2 * nbhd_size + 1, 2 * nbhd_size + 1))
                            # actual idx in data
                            for nbhd_x in range(x - nbhd_size, x + nbhd_size + 1):
                                for nbhd_y in range(y - nbhd_size, y + nbhd_size + 1):
                                    # boundary check
                                    if not (0 <= nbhd_x < data.shape[2] and 0 <= nbhd_y < data.shape[3]):
                                        continue
                                    # get features
                                    nbhd_feature[:, nbhd_x - (x - nbhd_size), nbhd_y - (y - nbhd_size)] = data[
                                                                                                          real_t,
                                                                                                          :,
                                                                                                          nbhd_x,
                                                                                                          nbhd_y]
                            nbhd_feature = nbhd_feature.flatten()

                            # last feature
                            last_feature = data[real_t - last_feature_num: real_t, :, x, y].flatten()

                            # hist feature
                            hist_feature = data[
                                           real_t - hist_feature_daynum * T: real_t: T, :,
                                           x, y].flatten()

                            feature_vec = np.concatenate((hist_feature, last_feature))
                            feature_vec = np.concatenate((feature_vec, nbhd_feature))

                            long_term_lstm_samples.append(feature_vec)
                        lstm_att_features[att_lstm_cnt].append(np.array(long_term_lstm_samples))
                    # label
                    labels.append(data[t, :, x, y].flatten())

        mmn.fit(self.dt)
        output_cnn_att_features = []
        for i in range(att_lstm_num):
            lstm_att_features[i]=mmn.transform(np.array(lstm_att_features[i]))
            for j in range(long_term_lstm_seq_len):
                cnn_att_features[i][j] = np.array(cnn_att_features[i][j])
                trans = mmn.transform(cnn_att_features[i][j])
                output_cnn_att_features.append(trans)

        cnnx = []
        for i in range(short_term_lstm_seq_len):
            cnnx.append([])
            cnn_features[i] = np.array(cnn_features[i])
            cnnx[i] = mmn.transform(cnn_features[i])

        for i in range(short_term_lstm_seq_len):
            cnn_features[i] = np.array(cnn_features[i])
        short_term_lstm_features = np.array(short_term_lstm_features)
        short_term_lstm_features = mmn.transform(short_term_lstm_features)
        labels = np.array(labels)
        labels = mmn.transform(labels)
        o1=np.array(output_cnn_att_features)
        o3=np.array(lstm_att_features)
        o4=np.array(cnnx)
        o6=np.array(short_term_lstm_features)
        o7=np.array(labels)
        print(o1.shape,o3.shape,o4.shape,o6.shape,o7.shape)
        return output_cnn_att_features, lstm_att_features, cnnx, short_term_lstm_features, labels, mmn


class models:
    def __init__(self):
        pass

    def Att_ConvLSTM(self, att_lstm_num, att_lstm_seq_len, lstm_seq_len, feature_vec_len, cnn_flat_size=128,
                     lstm_out_size=128,
                     nbhd_size=3, nbhd_type=2, output_shape=1, optimizer='adagrad',
                     loss='mse', metrics=[]):
        flatten_att_nbhd_inputs = [Input(shape=(nbhd_type, nbhd_size, nbhd_size),
                                         name="att_nbhd_volume_input_time_{0}_{1}".format(att + 1, ts + 1)) for ts in
                                   range(att_lstm_seq_len) for att in range(att_lstm_num)]
        att_nbhd_inputs = []
        for att in range(att_lstm_num):
            att_nbhd_inputs.append(flatten_att_nbhd_inputs[att * att_lstm_seq_len:(att + 1) * att_lstm_seq_len])

        att_lstm_inputs = [Input(shape=(att_lstm_seq_len, feature_vec_len), name="att_lstm_input_{0}".format(att + 1))
                           for att in range(att_lstm_num)]
        lstm_inputs = Input(shape=(lstm_seq_len, feature_vec_len), name="lstm_input")
        nbhd_inputs = [Input(shape=(nbhd_type, nbhd_size, nbhd_size), name="nbhd_volume_input_time_{0}".format(ts + 1))
                       for ts in range(lstm_seq_len)]

        nbhd_convs = [
            Conv2D(filters=64, kernel_size=(3, 3), padding="same", name="nbhd_convs_time0_{0}".format(ts + 1))(
                nbhd_inputs[ts]) for ts in range(lstm_seq_len)]
        nbhd_convs = [Activation("relu", name="nbhd_convs_activation_time0_{0}".format(ts + 1))(nbhd_convs[ts]) for ts
                      in range(lstm_seq_len)]
        nbhd_convs = [
            Conv2D(filters=64, kernel_size=(3, 3), padding="same", name="nbhd_convs_time1_{0}".format(ts + 1))(
                nbhd_convs[ts]) for ts in range(lstm_seq_len)]
        nbhd_convs = [Activation("relu", name="nbhd_convs_activation_time1_{0}".format(ts + 1))(nbhd_convs[ts]) for ts
                      in range(lstm_seq_len)]
        nbhd_convs = [
            Conv2D(filters=64, kernel_size=(3, 3), padding="same", name="nbhd_convs_time2_{0}".format(ts + 1))(
                nbhd_convs[ts]) for ts in range(lstm_seq_len)]
        nbhd_convs = [Activation("relu", name="nbhd_convs_activation_time2_{0}".format(ts + 1))(nbhd_convs[ts]) for ts
                      in range(lstm_seq_len)]
        nbhd_vecs = [Flatten(name="nbhd_flatten_time_{0}".format(ts + 1))(nbhd_convs[ts]) for ts in range(lstm_seq_len)]
        nbhd_vecs = [Dense(units=cnn_flat_size, name="nbhd_dense_time_{0}".format(ts + 1))(nbhd_vecs[ts]) for ts in
                     range(lstm_seq_len)]
        nbhd_vecs = [Activation("relu", name="nbhd_dense_activation_time_{0}".format(ts + 1))(nbhd_vecs[ts]) for ts in
                     range(lstm_seq_len)]

        nbhd_vec = Concatenate(axis=-1)(nbhd_vecs)
        nbhd_vec = Reshape(target_shape=(lstm_seq_len, cnn_flat_size))(nbhd_vec)
        lstm_input = Concatenate(axis=-1)([lstm_inputs, nbhd_vec])

        lstm = LSTM(units=lstm_out_size, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)(lstm_input)

        # attention part
        att_nbhd_convs = [[Conv2D(filters=64, kernel_size=(3, 3), padding="same",
                                  name="att_nbhd_convs_time0_{0}_{1}".format(att + 1, ts + 1))(att_nbhd_inputs[att][ts])
                           for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_convs = [[Activation("relu", name="att_nbhd_convs_activation_time0_{0}_{1}".format(att + 1, ts + 1))(
            att_nbhd_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]

        att_nbhd_convs = [[Conv2D(filters=64, kernel_size=(3, 3), padding="same",
                                  name="att_nbhd_convs_time1_{0}_{1}".format(att + 1, ts + 1))(att_nbhd_convs[att][ts])
                           for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_convs = [[Activation("relu", name="att_nbhd_convs_activation_time1_{0}_{1}".format(att + 1, ts + 1))(
            att_nbhd_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]

        att_nbhd_convs = [[Conv2D(filters=64, kernel_size=(3, 3), padding="same",
                                  name="att_nbhd_convs_time2_{0}_{1}".format(att + 1, ts + 1))(att_nbhd_convs[att][ts])
                           for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_convs = [[Activation("relu", name="att_nbhd_convs_activation_time2_{0}_{1}".format(att + 1, ts + 1))(
            att_nbhd_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]

        att_nbhd_vecs = [
            [Flatten(name="att_nbhd_flatten_time_{0}_{1}".format(att + 1, ts + 1))(att_nbhd_convs[att][ts]) for ts in
             range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_vecs = [[Dense(units=cnn_flat_size, name="att_nbhd_dense_time_{0}_{1}".format(att + 1, ts + 1))(
            att_nbhd_vecs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_vecs = [[Activation("relu", name="att_nbhd_dense_activation_time_{0}_{1}".format(att + 1, ts + 1))(
            att_nbhd_vecs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]

        att_nbhd_vec = [Concatenate(axis=-1)(att_nbhd_vecs[att]) for att in range(att_lstm_num)]
        att_nbhd_vec = [Reshape(target_shape=(att_lstm_seq_len, cnn_flat_size))(att_nbhd_vec[att]) for att in
                        range(att_lstm_num)]
        att_lstm_input = [Concatenate(axis=-1)([att_lstm_inputs[att], att_nbhd_vec[att]]) for att in
                          range(att_lstm_num)]

        att_lstms = [LSTM(units=lstm_out_size, return_sequences=True, dropout=0.1, recurrent_dropout=0.1,
                          name="att_lstm_{0}".format(att + 1))(att_lstm_input[att]) for att in range(att_lstm_num)]

        # compare
        att_low_level = [attention.Attention(method='cba')([att_lstms[att], lstm]) for att in range(att_lstm_num)]
        att_low_level = Concatenate(axis=-1)(att_low_level)
        att_low_level = Reshape(target_shape=(att_lstm_num, lstm_out_size))(att_low_level)

        att_high_level = LSTM(units=lstm_out_size, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)(
            att_low_level)
        att_high_level = attention.Attention(method='cba')([att_high_level,lstm])
        att_high_level = Reshape(target_shape=(att_lstm_num, lstm_out_size))(att_low_level)

        lstm_all = Concatenate(axis=-1)([att_high_level, lstm])
        lstm_all = Dense(units=output_shape)(lstm_all)
        pred_volume = Activation('tanh')(lstm_all)
        # inputs=[]
        # inputs.append(flatten_att_nbhd_inputs)
        # inputs.append(att_lstm_inputs)
        # inputs.append(nbhd_inputs)
        # inputs.append(lstm_inputs)
        inputs = flatten_att_nbhd_inputs + att_lstm_inputs + nbhd_inputs  + [lstm_inputs,]
        model = Model(inputs=inputs, outputs=pred_volume)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        model.summary()
        return model


def main(att_lstm_num=3, long_term_lstm_seq_len=3, short_term_lstm_seq_len=7, cnn_nbhd_size=3, nbhd_size=2,
         cnn_flat_size=128, \
         batch_size=64, max_epochs=100, validation_split=0.2, early_stop=EarlyStopping()):
    # model_hdf5_path = "./hdf5s/"

    modeler = models()
    sampler = file_loader()
    # training

    att_cnnx, att_x, cnnx, x, y, mmn = sampler.sample_Att_ConvLSTM(datatype='train', att_lstm_num=att_lstm_num, \
                                                                   long_term_lstm_seq_len=long_term_lstm_seq_len,
                                                                   short_term_lstm_seq_len=short_term_lstm_seq_len, \
                                                                   nbhd_size=nbhd_size, cnn_nbhd_size=cnn_nbhd_size)

    model = modeler.Att_ConvLSTM(att_lstm_num=att_lstm_num, att_lstm_seq_len=long_term_lstm_seq_len,
                                 lstm_seq_len=len(cnnx), feature_vec_len=x.shape[-1],
                                 cnn_flat_size=cnn_flat_size, nbhd_size=cnnx[0].shape[2], nbhd_type=cnnx[0].shape[1])

    model.fit(
        x=att_cnnx + att_x + cnnx + [x,],
        y=y,
        batch_size=batch_size, validation_split=validation_split, epochs=max_epochs, callbacks=[early_stop])

    att_cnnx, att_x, cnnx, x, y, mmn = sampler.sample_Att_ConvLSTM(datatype="test", att_lstm_num=att_lstm_num, \
                                                                   long_term_lstm_seq_len=long_term_lstm_seq_len,
                                                                   short_term_lstm_seq_len=short_term_lstm_seq_len, \
                                                                   nbhd_size=nbhd_size, cnn_nbhd_size=cnn_nbhd_size)
    y_pred = model.predict(x=att_cnnx + att_x + cnnx + [x,])
    # threshold = float(sampler.threshold) / sampler.config["volume_train_max"]
    # print("Evaluating threshold: {0}.".format(threshold))
    # (prmse, pmape), (drmse, dmape) = eval_lstm(y, y_pred, threshold)
    # print("Test on model {0}:\npickup rmse = {1}, pickup mape = {2}%\ndropoff rmse = {3}, dropoff mape = {4}%".format(
    #     model_name[2:], prmse, pmape * 100, drmse, dmape * 100))
    rmse = eval_lstm(y, y_pred)
    print('rmse (norm): %.6f rmse (real): %.6f' %
          (rmse, rmse * (mmn._max - mmn._min) / 2.))
    # currTime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    model.save_weights(os.path.join(
        'result', 'Att_ConvLSTM.h5'), overwrite=True)
    return


# early_stopping = EarlyStopping(monitor='val_rmse', patience=5, mode='min')
stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min')
batch_size = 64
max_epochs = 100

if __name__ == "__main__":
    main(batch_size=batch_size, max_epochs=max_epochs, early_stop=stop)
