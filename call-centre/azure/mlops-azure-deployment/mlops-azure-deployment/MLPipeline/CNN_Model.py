

import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten



# ## CNN
class CNN_model:

    def __init__(self, X_train, X_valid, X_test, Y_train, Y_valid, Y_test):
        epochs = 100
        batch = 256
        lr = 0.0003
        adam = optimizers.Adam(lr)

        X_train_series, X_valid_series = self.reshape_data(X_train, X_valid)

        model_cnn = self.model_arch(X_train_series, adam)

        self.model_train(X_train_series, X_valid_series, epochs, model_cnn, Y_train, Y_valid)

        self.model_predict(model_cnn, X_test, Y_test)

    def model_predict(self, model_cnn, X_test, Y_test):
        """
        model for prediction
        :param model_cnn:
        :param X_test:
        :param Y_test:
        :return:
        """
        cnn_pred = model_cnn.predict(X_test)
        # plot
        plt.figure(figsize=(20, 5))
        plt.plot(cnn_pred)
        plt.plot(Y_test.values, color='red')
        plt.savefig("output/"+"cnn.png")

    def model_train(self, X_train_series, X_valid_series, epochs, model_cnn, Y_train, Y_valid):
        """
        model for training
        :param X_train_series:
        :param X_valid_series:
        :param epochs:
        :param model_cnn:
        :param Y_train:
        :param Y_valid:
        :return:
        """
        cnn_history = model_cnn.fit(X_train_series, Y_train, validation_data=(X_valid_series, Y_valid), epochs=epochs,
                                    verbose=2)

    def model_arch(self, X_train_series, adam):
        """
        model architecture
        :param X_train_series:
        :param adam:
        :return:
        """
        model_cnn = Sequential()
        model_cnn.add(Conv1D(filters=64, kernel_size=2, activation='relu',
                             input_shape=(X_train_series.shape[1], X_train_series.shape[2])))
        model_cnn.add(MaxPooling1D(pool_size=2))
        model_cnn.add(Flatten())
        model_cnn.add(Dense(50, activation='relu'))
        model_cnn.add(Dense(1))
        model_cnn.compile(loss='mse', optimizer=adam)
        model_cnn.summary()
        return model_cnn

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
