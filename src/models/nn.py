from metric import *
from data import *
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Masking, Dropout, LSTM


class NN:
    def __init__(self, model="cnn", feature_dims=4, lr=0.01, epochs=500, special_value=-10):
        if model == "lstm":
            self.model = LstmModel(feature_dims, special_value)
            self.callbacks = keras.callbacks.EarlyStopping(monitor="loss", patience=50, verbose=1)
            self.opt = keras.optimizers.Adam(learning_rate=lr)
            self.epoch = epochs

    def forward(self, x_train, y_train, batch_size=32):

        self.model.compile(optimizer=self.opt, loss="categorical_crossentropy", metrics=['acc', f1_macro, precision_macro, recall_macro])
        return self.model.fit(x_train, y_train, batch_size=batch_size, epochs=self.epoch, callbacks=self.callbacks, validation_split=0.1, verbose=1)

    def evaluate_model(self, x_test, y_test):
        loss, accuracy, f1_score, precision, recall = self.model.evaluate(x_test, y_test, verbose=0)
        print("Test loss", loss)
        print("Test accuracy", accuracy)
        print("Test precision", precision)
        print("Test recall", recall)
        print("Test f1_score", f1_score)


class LstmModel:
    def __init__(self, lstm_unit, feature_dims, special_value):
        self.model = Sequential()
        self.model.add(Masking(mask_value=special_value, input_shape=(max_seq_len, feature_dims)))
        self.model.add(LSTM(lstm_unit))
        self.model.add(Dropout(drop_out))
        self.model.add(Dense(dense, activation=act))
        self.model.add(Dense(n_out, activation='softmax'))


if __name__ == '__main__':

    path = "/data/home/masoumeh/Data/"
    df_data = read_csv_file(path+"dataset_big_clean.csv")
    X_data, y_data = create_data_for_training(df_data, with_pad=True, model="nn")
    X_train, y_train, X_test, y_test = split_train_test(X_data, y_data, shuffle=True)
    ytrain = keras.utils.to_categorical(y_train)
    ytest = keras.utils.to_categorical(y_test)

    # hyper parameters
    lstm_unit = 300
    max_seq_len = X_data[0].shape[0]  # 30??
    feature_dim = X_data[0].shape[1]  # 4
    drop_out = 0.1
    dense = 100
    act = "relu"
    n_out = len(np.unique(y_train))  # 3
    masking_value = -10

    hyper_p = (lstm_unit, max_seq_len, feature_dim, drop_out, dense, act, n_out, masking_value)
    lstm_model = NN(model="lstm", hyper_params=hyper_p)
    lstm_model.forward(X_train, ytrain, epochs=500, lr=0.01, batch_size=32)
    lstm_model.evaluate_model(X_test, ytest)