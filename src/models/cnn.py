import numpy as np
from utils import load_dataset
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from plot import plot_confusion_matrix, plot_model_loss
from metric import compute_acc_prec_rec_f1
from sklearn.metrics import classification_report


def make_1DCNN(input_shape, num_classes, drop_out, special_value):
    """Build a 1D convolutional neural network model."""
    input_layer = keras.layers.Input(input_shape)
    masking = keras.layers.Masking(mask_value=special_value, input_shape=input_shape)(input_layer)

    x = keras.layers.Conv1D(filters=64, kernel_size=3,  padding='same')(masking)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling1D(pool_size=2)(x)

    x = keras.layers.Conv1D(filters=64, kernel_size=3,  padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling1D(pool_size=2)(x)

    x = keras.layers.Conv1D(filters=128, kernel_size=3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling1D(pool_size=2)(x)
    # #
    x = keras.layers.Conv1D(filters=256, kernel_size=3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling1D(pool_size=2)(x)

    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(units=512, activation="relu")(x)
    x = keras.layers.Dropout(drop_out)(x)
    output_layer = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.models.Model(inputs=input_layer, outputs=output_layer)


def make_1DBiCNN(input_shape, num_classes, drop_out, special_value):
    """Build a 1D convolutional neural network model."""
    input_layer = keras.layers.Input(input_shape)
    masking = keras.layers.Masking(mask_value=special_value, input_shape=input_shape)(input_layer)

    x = keras.layers.Conv1D(filters=64, kernel_size=3, padding='same')(masking)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling1D(pool_size=2)(x)

    x = keras.layers.Conv1D(filters=64, kernel_size=3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling1D(pool_size=2)(x)

    x = keras.layers.Conv1D(filters=128, kernel_size=3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling1D(pool_size=2)(x)
    # # #
    x = keras.layers.Conv1D(filters=256, kernel_size=3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling1D(pool_size=2)(x)

    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(units=512, activation="relu")(x)
    x = keras.layers.Dropout(drop_out)(x)
    output_layer = keras.layers.Dense(num_classes, activation="sigmoid")(x)
    return keras.models.Model(inputs=input_layer, outputs=output_layer)


def train_model(model, x_train, y_train, x_test, y_test, lr, epochs, batch_size, loss):
    initial_learning_rate = lr
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )
    model.compile(
        loss=loss,
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=['acc'])

    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
    return model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                     verbose=1, validation_data=(x_test, y_test), callbacks=early_stopping_cb)


def train(xtrain, ytrain, xtest, ytest,
        input_shape=None,
        n_classes=3,
        drop_out=0.1,
        special_value=-98745.0,
        lr=0.0001,
        epochs=10,
        batch_size=64,
        loss_func="categorical_crossentropy",
        task='multi'
        ):

    if task == 'multi':
        model = make_1DCNN(input_shape=input_shape,
                             num_classes=n_classes,
                             drop_out=drop_out,
                             special_value=special_value)
    else:
        model = make_1DBiCNN(input_shape=input_shape,
                           num_classes=n_classes,
                           drop_out=drop_out,
                           special_value=special_value)

    history = train_model(model=model, x_train=xtrain, y_train=ytrain, x_test=xtest, y_test=ytest,
                          lr=lr, epochs=epochs, batch_size=batch_size, loss=loss_func)
    return model, history


def predict(model, xtest):
    y_pred = np.argmax(model.predict(xtest), axis=1)
    return y_pred


def error_analysis(history,
                   y_true,
                   y_pred,
                   model_name,
                   set_name,
                   task,
                   iters,
                   path_to_save,
                   target_names):

    # plot history
    plot_model_loss(history=history,
                    title='Loss on the Train and Test Datasets',
                    dataset=set_name,
                    model_name=model_name,
                    task=task,
                    save=True,
                    show=False,
                    path=path_to_save+'loss/',
                    filename='plot_loss_'+set_name+'_'+model_name+"_"+str(iters)+'_'+task)

    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    print(cm)
    plot_confusion_matrix(cm,
                          normalize=False,
                          target_names=target_names,
                          title="Confusion Matrix",
                          dataset=set_name,
                          model_name=model_name,
                          task=task,
                          save=True,
                          path=path_to_save+'confusion_matrix/',
                          filename='confusion_matrix_'+set_name+'_'+model_name+"_"+str(iters)+'_'+task)

    plot_confusion_matrix(cm,
                          normalize=True,
                          target_names=target_names,
                          title="Confusion Matrix, Normalized",
                          dataset=set_name,
                          model_name=model_name,
                          task=task,
                          save=True,
                          path=path_to_save+'confusion_matrix/',
                          filename='confusion_matrix_normalized'+set_name+'_'+model_name+"_"+str(iters)+'_'+task)
    print('Done!')


def run(fname_prefix='set_1',
        model_dim='1d',
        iters=1,
        metric_avg='macro',
        task='multi',
        loss_func="categorical_crossentropy",
        n_classes=3,
        target_names=None,
        model_name='cnn1d',
        n_features=4,
        pad_value=-98745.0,
        drop_out=0.1,
        epochs=100,
        batch_size=64,
        lr=0.0001,
        path_to_read=None,
        path_to_write=None
        ):

    xtrain, ytrain, xtest, ytest, max_seq_len = load_dataset(fname_prefix, path_to_read)
    y_train = keras.utils.to_categorical(ytrain)
    y_test = keras.utils.to_categorical(ytest)

    if model_dim == '2d':
        print('2D model')
        xtrain = xtrain.reshape(xtrain.shape[0], max_seq_len, n_features, 1)
        xtest = xtest.reshape(xtest.shape[0], max_seq_len, n_features, 1)

    print('xtrain.shape = ', xtrain.shape)
    print('ytrain.shape = ', ytrain.shape)
    print('xtest.shape = ', xtest.shape)
    print('ytest.shape = ', ytest.shape)

    print('-----------------------------')
    print('max seq length = ', max_seq_len)

    model, history = train(xtrain, y_train, xtest, y_test,
                           input_shape=(max_seq_len, n_features),
                           n_classes=n_classes,
                           drop_out=drop_out,
                           special_value=pad_value,
                           lr=lr,
                           epochs=epochs,
                           batch_size=batch_size,
                           loss_func=loss_func,
                           task=task)

    predictions = predict(model, xtest)
    error_analysis(history=history,
                   y_true=ytest,
                   y_pred=predictions,
                   model_name=model_name,
                   set_name=fname_prefix,
                   task=task,
                   iters=iters,
                   path_to_save=path_to_write,
                   target_names=target_names)
    print('-------------------------------------------')
    print()
    report = classification_report(ytest, predictions, target_names=target_names)
    print(report)
    print('------------------------------------------')
    # print(model.summary())
    return compute_acc_prec_rec_f1(y_true=ytest, y_pred=predictions, average=metric_avg)




# model_name = 'nn_cnn'
# task = ['multi', ]
# loss_func = "categorical_crossentropy"
# target_classes = ['deictic', 'sequential', 'demarcative']
# where_to_read = "/mnt/shared/people/masoumeh/MA/data/nn/"
# n_classes = 3
# n_features = 4
# # fname_prefix = 'set_13'
# batch_size = 64
# drop_out = 0.1
# epochs = 100
# lr = 0.0001
# model_dim = '1d'
# metric_avg = 'macro'
# pad_value = -98745.0
# where_to_write = "/mnt/shared/people/masoumeh/MA/results/plots/"
#
# # for binary classification
# where_to_read = "/mnt/shared/people/masoumeh/MA/data/nn/binary/"
# loss_func = "binary_crossentropy"
# target_classes = ['Others', 'demarcative']
# task='bi_'+target_classes[1]
# # target_classes = ['Others', 'sequential']
# # target_classes = ['Others', 'deictic']
# n_classes=2




