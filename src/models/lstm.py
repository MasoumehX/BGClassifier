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


def evaluate_model(x_test, y_test):
    model = keras.models.load_model("best_model.h5")
    loss, accuracy, f1_score, precision, recall = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss", loss)
    print("Test accuracy", accuracy)
    print("Test precision", precision)
    print("Test recall", recall)
    print("Test f1_score", f1_score)


if __name__ == '__main__':

    # On Server
    path = "/data/home/masoumeh/Data/"
    df_data = read_csv_file(path+"dataset_big_clean.csv")
    X_data, y_data = create_data_for_training(df_data, with_pad=True, model="nn")
    X_train, y_train, X_test, y_test = split_train_test(X_data, y_data, shuffle=True)
    ytrain = keras.utils.to_categorical(y_train)
    ytest = keras.utils.to_categorical(y_test)

    # features = ["x", "y", "poi", "frame"]
    max_seq_len = X_data[0].shape[0]  # 30??
    feature_dim = X_data[0].shape[1]  # 4
    n_classes = len(np.unique(y_train))  # 3
    lstm_model = make_model(lstm_unit=300, max_seq_len=max_seq_len, dimension=feature_dim, drop_out=0.1, lr=0.01,
                            dense=200, n_out=n_classes, special_value=-10, act='relu')
    output = train_model(lstm_model, X_train, ytrain, epochs=300, batch_size=64)

    # evaluate the model on test data
    evaluate_model(X_test, ytest)
