import numpy as np
from tensorflow import keras
from metric import *
from utils import read_csv_file, split_train_test
from data import create_data_for_training


def make_model(input_shape, num_classes, max_seq_len, n_features=4, n_filters=64, kernel_size=3, reg_lr=0.001, padding='same', special_value=-10, ):

    input_layer = keras.layers.Input(input_shape)
    masking = keras.layers.Masking(mask_value=special_value, input_shape=(max_seq_len, n_features))(input_layer)
    conv1 = keras.layers.Conv1D(filters=n_filters, kernel_size=kernel_size, kernel_regularizer=keras.regularizers.l2(l=reg_lr), padding=padding)(masking)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=n_filters, kernel_size=kernel_size, kernel_regularizer=keras.regularizers.l2(l=reg_lr), padding=padding)(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=n_filters, kernel_size=kernel_size, kernel_regularizer=keras.regularizers.l2(l=reg_lr), padding=padding)(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)
    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)
    return keras.models.Model(inputs=input_layer, outputs=output_layer)


def train_model(model, x_train, y_train, epochs=500, lr=0.01, batch_size=32, save_model_name="best_model.h5"):
    callbacks = [
        keras.callbacks.ModelCheckpoint(save_model_name, save_best_only=True, monitor="loss"),
        keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=20, min_lr=0.0001),
        keras.callbacks.EarlyStopping(monitor="loss", patience=50, verbose=1),
    ]
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['acc',f1_macro,precision_macro, recall_macro])
    return model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks, validation_split=0.1, verbose=1)


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
    df_data = read_csv_file(path+"dataset_big_clean.csv")

    features = ["x", "y", "poi", "frame"]
    X_data, y_data = create_data_for_training(df_data, features=features, with_pad=True, model="nn")
    X_train, y_train, X_test, y_test = split_train_test(X_data, y_data, test_size=0.3, shuffle=True)

    # remove data.x == 0
    ytrain = keras.utils.to_categorical(y_train)
    ytest = keras.utils.to_categorical(y_test)

    n_classes = len(np.unique(y_train))

    # create a model
    max_seq_len = X_data[0].shape[0]
    # cnn_model = make_model(input_shape=X_train.shape[1:], num_classes=n_classes, reg_lr=0.001, n_filters=32, kernel_size=3, padding='same')
    cnn_model = make_model(X_train.shape[1:], num_classes=n_classes, max_seq_len=max_seq_len, n_features=4, n_filters=64,
                           kernel_size=3, reg_lr=0.01, padding='same', special_value=-10)

    # TODO: using iters works really well!!!!

    for i in range(10):
        # Training the model
        train_model(model=cnn_model, x_train=X_train, y_train=ytrain, lr=0.001, epochs=500, batch_size=64, save_model_name="cnn_n_filters_32.h5")

        # model evaluation on test set
        evaluate_model(cnn_model, X_test, ytest)



