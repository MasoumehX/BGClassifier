from tensorflow import keras
from metric import *
from utils import read_csv_file
from data import create_data_for_training
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, Masking


def make_1Dmodel(input_shape, num_classes, max_seq_len, n_features=4, n_filters=64, kernel_size=3, padding='same', special_value=-10):

    input_layer = keras.layers.Input(input_shape)
    masking = keras.layers.Masking(mask_value=special_value, input_shape=(max_seq_len, n_features))(input_layer)
    conv1 = keras.layers.Conv1D(filters=n_filters, kernel_size=kernel_size, padding=padding)(masking)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=n_filters, kernel_size=kernel_size, padding=padding)(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=n_filters, kernel_size=kernel_size, padding=padding)(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)
    before_out = keras.layers.Dropout(0.3)(gap)
    output_layer = keras.layers.Dense(num_classes, activation="softmax")(before_out)
    return keras.models.Model(inputs=input_layer, outputs=output_layer)


def make_2Dmodel(max_seq_len, num_classes, n_features=4, special_value=-10):
    """Build a 2D convolutional neural network model."""

    model = Sequential()

    model.add(Masking(mask_value=special_value, input_shape=(max_seq_len, n_features, 1)))

    # convolutional layer
    model.add(Conv2D(50, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=(max_seq_len, n_features, 1)))

    # convolutional layer
    model.add(Conv2D(75, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(125, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # # flatten output of conv
    # model.add(Flatten())
    #
    # # hidden layer
    # model.add(Dense(500, activation='relu'))
    # model.add(Dropout(0.4))
    # model.add(Dense(250, activation='relu'))
    # model.add(Dropout(0.3))
    # output layer
    model.add(Dense(num_classes, activation='softmax'))
    return model


def train_model(model, x_train, y_train, x_test, y_test, epochs=500, lr=0.01, batch_size=32, save_model_name="best_model.h5"):
    callbacks = [
        keras.callbacks.ModelCheckpoint(save_model_name, save_best_only=True, monitor="val_loss"),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
    ]
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['acc',f1_m,precision_m, recall_m])
    return model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks, validation_data=(x_test, y_test), verbose=1)


def evaluate_model(model, x_test, y_test):
    loss, accuracy, f1_score, precision, recall = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss", loss)
    print("Test accuracy", accuracy)
    print("Test precision", precision)
    print("Test recall", recall)
    print("Test f1_score", f1_score)


if __name__ == '__main__':

    # On Server
    path = "/data/home/masoumeh/Data/"
    df_train = read_csv_file(path+"df_train.csv")
    df_test = read_csv_file(path+"df_test.csv")

    feature_cols = ["x", "y", "poi", "time"]

    # Create data for Neural Network models : CNN
    xtrain, ytrain, max_seq_len = create_data_for_training(df_train, with_pad=True, features=feature_cols, model="nn")
    xtest, ytest, _ = create_data_for_training(df_test, with_pad=True, max_seq_len=max_seq_len, features=feature_cols, model="nn")

    # remove data.x == 0
    ytrain = keras.utils.to_categorical(ytrain)
    ytest = keras.utils.to_categorical(ytest)

    n_classes = 3

    # create a model
    cnn_model = make_1Dmodel(xtrain.shape[1:], num_classes=n_classes, max_seq_len=max_seq_len, n_features=4, n_filters=64,
                           kernel_size=3, padding='same', special_value=-10)

    # reshape the data for 2D CNN
    # x_train = xtrain.reshape(xtrain.shape[0], 3044, 4, 1)
    # x_test = xtest.reshape(xtest.shape[0], 3044, 4, 1)
    # cnn_model = make_2Dmodel(num_classes=n_classes, max_seq_len=max_seq_len, n_features=4, special_value=-10)

    # Training the model
    for i in range(10):
        train_model(model=cnn_model, x_train=xtrain, y_train=ytrain, x_test=xtest, y_test=ytest, lr=0.002, epochs=100, batch_size=32,
                    save_model_name="cnn_n_filters_32.h5")
        # model evaluation on test set
        evaluate_model(cnn_model, xtest, ytest)
        evaluate_model(cnn_model, xtrain, ytrain)

    # cnn_model.summary()



