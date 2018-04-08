# coding=utf-8
from __future__ import print_function
import os

from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, Activation, Flatten, Concatenate, Reshape, LSTM, Dense
from mydeepst.datasets.load_data import load_h5data
from mydeepst.preprocessing.minmax_normalization import MinMaxNormalization
import numpy as np
import tensorflow as tf

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

T = 48
days_test = 7
len_test = T * days_test


class CustomStopper(EarlyStopping):
    # add argument for starting epoch
    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', start_epoch=40):
        super().__init__(monitor=monitor, min_delta=min_delta, patience=patience, verbose=verbose, mode=mode)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)


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
    def __init__(self, config_path="data.json"):
        self.isVolumeLoaded = False
        self.isFlowLoaded = False

    def load_volume(self):
        # shape (timeslot_num, type=1, x_num, y_num)
        self.ts, self.dt = load_h5data('data.h5')
        self.ts_train, self.volume_train = self.ts[:-len_test], self.dt[:-len_test]
        self.ts_test, self.volume_test = self.ts[-len_test:], self.dt[-len_test:]

    def sample_ConvLSTM(self, datatype, short_term_lstm_seq_len=8, \
                        cnn_nbhd_size=3):
        if self.isVolumeLoaded is False:
            self.load_volume()

        if datatype == "train":
            data = self.volume_train
        elif datatype == "test":
            data = self.volume_test
        else:
            print("Please select **train** or **test**")
            raise Exception

        cnn_features = []
        for i in range(short_term_lstm_seq_len):
            cnn_features.append([])
        labels = []
        time_start = short_term_lstm_seq_len
        time_end = data.shape[0]
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
                    # label
                    labels.append(data[t, :, x, y].flatten())
        mmn.fit(self.dt)
        x = []
        for i in range(short_term_lstm_seq_len):
            x.append([])
            cnn_features[i] = np.array(cnn_features[i])
            x[i] = mmn.transform(cnn_features[i])
        labels = np.array(labels)
        y = mmn.transform(labels)
        return x, y, mmn


class models:
    def __init__(self):
        pass

    def ConvLSTM(self, lstm_seq_len, cnn_flat_size=128, lstm_out_size=128,
                 nbhd_size=3, nbhd_type=1, output_shape=1, optimizer='adagrad', loss='mse',
                 metrics=[]):
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
        lstm_input = Reshape(target_shape=(lstm_seq_len, cnn_flat_size))(nbhd_vec)
        lstm = LSTM(units=lstm_out_size, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)(lstm_input)
        lstm = Dense(units=output_shape)(lstm)
        pred_volume = Activation('tanh')(lstm)
        inputs = nbhd_inputs
        model = Model(inputs=inputs, outputs=pred_volume)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        model.summary()
        return model


def main(cnn_flat_size=128,
         batch_size=64, max_epochs=100, validation_split=0.2, early_stop=EarlyStopping()):
    # model_hdf5_path = "./hdf5s/"

    modeler = models()
    sampler = file_loader()
    # training

    x, y, mmn = sampler.sample_ConvLSTM(datatype='train')

    model = modeler.ConvLSTM(lstm_seq_len=len(x), cnn_flat_size=cnn_flat_size, nbhd_size=x[0].shape[2],
                             nbhd_type=x[0].shape[1])

    model.fit(
        x=x,
        y=y,
        batch_size=batch_size, validation_split=validation_split, epochs=max_epochs, callbacks=[early_stop])

    x, y, mmn = sampler.sample_ConvLSTM(datatype="test")
    y_pred = model.predict(x=x)
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
        'result', 'ConvLSTM.h5'), overwrite=True)
    return


# early_stopping = EarlyStopping(monitor='val_rmse', patience=5, mode='min')
stop = CustomStopper(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min', start_epoch = 40)
batch_size = 64
max_epochs = 100

if __name__ == "__main__":
    main(batch_size=batch_size, max_epochs=max_epochs, early_stop=stop)
