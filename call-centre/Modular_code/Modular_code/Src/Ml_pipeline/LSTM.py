

import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten


### LSTM - build a class for lstm model
class LSTM_model:

    def __init__(self, X_train, X_valid, X_test, Y_train, Y_valid, Y_test):
       # define the parameters
        epochs = 100
        batch = 256
        lr = 0.0003
        adam = optimizers.Adam(lr)

        X_train_series, X_valid_series = self.reshape_data(X_train, X_valid)

        model_lstm = self.model_arch(X_train_series, adam)

        model_lstm = self.model_train(X_train_series, X_valid_series, epochs, model_lstm, Y_train, Y_valid)

        self.model_predict(model_lstm, X_test, Y_test)


    def model_predict(self, model_lstm, X_test, Y_test):
        """
        model prediction
        :param model_lstm:
        :param X_test:
        :param Y_test:
        :return:
        """
        print("Predict")
        X_test_series = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        lstm_pred = model_lstm.predict(X_test_series)
        # plot
        plt.figure(figsize=(20, 5))
        plt.plot(lstm_pred)
        plt.plot(Y_test.values, color='red')
        plt.savefig("../Output/plots/"+"lstm.png")


    def model_train(self, X_train_series, X_valid_series, epochs, model_lstm, Y_train, Y_valid):
        """
        model training
        :param X_train_series:
        :param X_valid_series:
        :param epochs:
        :param model_lstm:
        :param Y_train:
        :param Y_valid:
        :return:
        """
        lstm_history = model_lstm.fit(X_train_series, Y_train, validation_data=(X_valid_series, Y_valid), epochs=epochs,
                                      verbose=2)
        return model_lstm


    def model_arch(self, X_train_series, adam):
        """
        model architecture
        :param X_train_series:
        :param adam:
        :return:
        """
        model_lstm = Sequential()
        model_lstm.add(LSTM(50, activation='relu', input_shape=(X_train_series.shape[1], X_train_series.shape[2])))
        model_lstm.add(Dense(1))
        model_lstm.compile(loss='mse', optimizer=adam)
        model_lstm.summary()
        return model_lstm


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
