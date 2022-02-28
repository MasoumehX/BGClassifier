from data import *
from metric import *
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Masking, Dropout, LSTM


def make_model(lstm_unit, max_seq_len, dimension, drop_out, dense, act, n_out=3, special_value=-10):
    model = Sequential()
    model.add(Masking(mask_value=special_value, input_shape=(max_seq_len, dimension)))
    model.add(LSTM(lstm_unit))
    model.add(Dropout(drop_out))
    model.add(Dense(dense, activation=act))
    model.add(Dense(n_out, activation='softmax'))
    return model


def train_model(model, x_train, y_train, lr=0.01, epochs=500, batch_size=32):
    callbacks = [
        keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True, monitor="loss"),
        keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=20, min_lr=0.0001),
        keras.callbacks.EarlyStopping(monitor="loss", patience=50, verbose=1),
    ]
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc', f1_m, precision_m, recall_m])
    return model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks, validation_split=0.1, verbose=1)


def evaluate_model(model, x_test, y_test):
    loss, accuracy, f1_score, precision, recall = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss", loss)
    print("Test accuracy", accuracy)
    print("Test precision", precision)
    print("Test recall", recall)
    print("Test f1_score", f1_score)


def load_data():
    # On Server
    path = "/data/home/masoumeh/Data/"
    data_train = read_csv_file(path + "train.csv")
    data_test = read_csv_file(path + "test.csv")

    #
    # print("total gestures per (semantic) class for Train:")
    # print(" --- total gestures per `demarcative` class:", y_train.tolist().count(2))
    # print(" --- total gestures per `deictic` class:", y_train.tolist().count(0))
    # print(" --- total gestures per `sequential` class:", y_train.tolist().count(1))
    # print("total gestures in train: ", y_train.shape[0])
    #
    # print('-'*50)
    #
    # print("total gestures per (semantic) class for Test:")
    # print(" --- total gestures per `demarcative` class:", y_test.tolist().count(2))
    # print(" --- total gestures per `deictic` class:", y_test.tolist().count(0))
    # print(" --- total gestures per `sequential` class:", y_test.tolist().count(1))
    # print("total gestures in train: ", y_test.shape[0])

    return data_train["features"], data_train["label"], data_test["features"], data_test["label"]


def run():
    xtrain, ytrain, xtest, ytest = load_data()

    ytrain = keras.utils.to_categorical(ytrain)
    ytest = keras.utils.to_categorical(ytest)

    feature_dim = xtrain[0].shape[1]
    max_seq_len = xtrain[0].shape[0]
    n_classes = ytrain.unique().shape[0]

    lstm_model = make_model(lstm_unit=300, max_seq_len=max_seq_len, dimension=feature_dim, drop_out=0.1,
                            dense=200, n_out=n_classes, special_value=-10, act='relu')
    train_model(lstm_model, xtrain, ytrain, epochs=300, batch_size=64, lr=0.002)

    # evaluate the model on test data
    evaluate_model(lstm_model, xtest, ytest)
    make_model()

