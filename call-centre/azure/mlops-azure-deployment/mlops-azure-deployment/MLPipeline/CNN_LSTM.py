

import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten



import math

class CNN_LSTM:

    def __init__(self, X_train, X_valid, X_test, Y_train, Y_valid, Y_test):

        epochs = 100
        batch = 256
        lr = 0.0003
        adam = optimizers.Adam(lr)

        X_train_series, X_valid_series = self.reshape_data(X_train, X_valid)

        X_train_series_sub, X_valid_series_sub = self.set_data(X_train_series, X_valid_series)

        model_cnn_lstm = self.model_arch(X_train_series_sub, adam)

        self.model_train(X_train_series_sub, X_valid_series_sub, Y_train, Y_valid, epochs, model_cnn_lstm)

        self.model_predict(X_test, Y_test, model_cnn_lstm)

    def model_predict(self, X_test, Y_test, model_cnn_lstm):
        """
        Model for prediction
        :param X_test:
        :param Y_test:
        :param model_cnn_lstm:
        :return:
        """
        X_test_series = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        X_test_series_sub = X_test_series.reshape((X_test_series.shape[0], 1, 5, 1))
        cnn_lstm_pred = model_cnn_lstm.predict(X_test_series_sub)
        # plot
        plt.figure(figsize=(20, 5))
        plt.plot(cnn_lstm_pred)
        plt.plot(Y_test.values, color='red')
        plt.savefig("output/"+"cnn_lstm.png")

    def model_train(self, X_train_series_sub, X_valid_series_sub, Y_train, Y_valid, epochs, model_cnn_lstm):
        """
        Model for train
        :param X_train_series_sub:
        :param X_valid_series_sub:
        :param Y_train:
        :param Y_valid:
        :param epochs:
        :param model_cnn_lstm:
        :return:
        """
        cnn_lstm_history = model_cnn_lstm.fit(X_train_series_sub, Y_train,
                                              validation_data=(X_valid_series_sub, Y_valid), epochs=epochs, verbose=2)

    def model_arch(self, X_train_series_sub, adam):
        """
        Setting the model Arch
        :param X_train_series_sub:
        :param adam:
        :return:
        """
        model_cnn_lstm = Sequential()
        model_cnn_lstm.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(
        None, X_train_series_sub.shape[2], X_train_series_sub.shape[3])))
        model_cnn_lstm.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model_cnn_lstm.add(TimeDistributed(Flatten()))
        model_cnn_lstm.add(LSTM(50, activation='relu'))
        model_cnn_lstm.add(Dense(1))
        model_cnn_lstm.compile(loss='mse', optimizer=adam)
        return model_cnn_lstm

    def set_data(self, X_train_series, X_valid_series):
        """
        reshaping the data
        :param X_train_series:
        :param X_valid_series:
        :return:
        """
        X_train_series_sub = X_train_series.reshape((X_train_series.shape[0], 1, 5, 1))
        X_valid_series_sub = X_valid_series.reshape((X_valid_series.shape[0], 1, 5, 1))
        print('Train set shape', X_train_series_sub.shape)
        print('Validation set shape', X_valid_series_sub.shape)
        return X_train_series_sub, X_valid_series_sub

    def reshape_data(self, X_train, X_valid):
        """
        reshaping the data
        :param X_train:
        :param X_valid:
        :return:
        """
        X_train_series = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_valid_series = X_valid.reshape((X_valid.shape[0], X_valid.shape[1], 1))
        print('Train set shape', X_train_series.shape)
        print('Validation set shape', X_valid_series.shape)
        return X_train_series, X_valid_series

