

import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten


# ## Neural Networks (Dense)

class MLP:

    def __init__(self, X_train, X_valid, X_test, Y_train, Y_valid, Y_test):
        epochs = 100
        batch = 256
        lr = 0.0003
        adam = optimizers.Adam(lr)

        model_mlp = self.model_arch(adam, X_train)

        self.model_train(epochs, model_mlp, X_train, Y_train, X_valid, Y_valid)

        self.model_predict(model_mlp, X_test, Y_test)

    def model_predict(self, model_mlp, X_test, Y_test):
        """
        model prediction
        :param model_mlp:
        :param X_test:
        :param Y_test:
        :return:
        """
        mlp_pred = model_mlp.predict(X_test)
        # plot
        plt.figure(figsize=(20, 5))
        plt.plot(mlp_pred)
        plt.plot(Y_test.values, color='red')
        plt.savefig("output/"+"mlp.png")

    def model_train(self, epochs, model_mlp, X_train, Y_train, X_valid, Y_valid):
        """
        model training
        :param epochs:
        :param model_mlp:
        :param X_train:
        :param Y_train:
        :param X_valid:
        :param Y_valid:
        :return:
        """
        mlp_history = model_mlp.fit(X_train, Y_train, validation_data=(X_valid, Y_valid),
                                    epochs=epochs, verbose=2)

    def model_arch(self, adam, X_train):
        """
        model architecture
        :param adam:
        :param X_train:
        :return:
        """
        model_mlp = Sequential()
        model_mlp.add(Dense(100, activation='relu', input_dim=X_train.shape[1]))
        model_mlp.add(Dense(1))
        model_mlp.compile(loss='mse', optimizer=adam)
        model_mlp.summary()
        return model_mlp
