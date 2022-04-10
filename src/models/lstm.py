from utils import load_dataset
from plot import plot_confusion_matrix, plot_model_loss
import numpy as np
from numpy import mean
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from metric import compute_acc_prec_rec_f1


def make_LSTM(input_shape, lstm_units, num_classes, drop_out, special_value):
    input_layer = keras.layers.Input(input_shape)
    masking = keras.layers.Masking(mask_value=special_value, input_shape=input_shape)(input_layer)
    x = keras.layers.LSTM(lstm_units)(masking)
    x = keras.layers.Dropout(drop_out)(x)
    x = keras.layers.Dense(100, activation="relu")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(input_layer, outputs)


def train(xtrain, ytrain, xtest, ytest,
        input_shape=None,
        n_classes=3,
        drop_out=0.1,
        special_value=-98745.0,
        lr=0.0001,
        epochs=10,
        batch_size=64,
        loss_func="categorical_crossentropy",
        lstm_units=300
        ):

    model = make_LSTM(input_shape=input_shape,
                      lstm_units=lstm_units,
                       num_classes=n_classes,
                       drop_out=drop_out,
                       special_value=special_value)

    history = train_model(model=model, x_train=xtrain, y_train=ytrain, x_test=xtest, y_test=ytest,
                          lr=lr, epochs=epochs, batch_size=batch_size, loss=loss_func)
    return model, history


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


def predict(model, xtest):
    y_pred = np.argmax(model.predict(xtest), axis=1)
    return y_pred


def error_analysis(history, y_true, y_pred, model_name, set_name, task, iters, path_to_save):
    print(" -----------------------Error Analysis---------------------------")

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
                          target_names=['deictic', 'sequential', 'demarcative'],
                          title="Confusion Matrix",
                          dataset=set_name,
                          model_name=model_name,
                          task=task,
                          save=True,
                          path=path_to_save+'confusion_matrix/',
                          filename='confusion_matrix_'+set_name+'_'+model_name+"_"+str(iters)+'_'+task)

    plot_confusion_matrix(cm,
                          normalize=True,
                          target_names=['deictic', 'sequential', 'demarcative'],
                          title="Confusion Matrix, Normalized",
                          dataset=set_name,
                          model_name=model_name,
                          task=task,
                          save=True,
                          path=path_to_save+'confusion_matrix/',
                          filename='confusion_matrix_normalized'+set_name+'_'+model_name+"_"+str(iters)+'_'+task)
    print('Done!')


def run(fname_prefix='set_1', iters=1, metric_avg='macro', lstm_units=2):

        xtrain, ytrain, xtest, ytest, max_seq_len = load_dataset(fname_prefix)
        y_train = keras.utils.to_categorical(ytrain)
        y_test = keras.utils.to_categorical(ytest)

        print('xtrain.shape = ', xtrain.shape)
        print('ytrain.shape = ', ytrain.shape)
        print('xtest.shape = ', xtest.shape)
        print('ytest.shape = ', ytest.shape)

        print('-----------------------------')
        print('max seq length = ', max_seq_len)

        # hyper parameters
        n_features = 3
        n_classes = 3
        pad_value = -98745.0
        drop_out = 0.1
        epochs = 5
        batch_size = 64
        lr = 0.001
        loss_func = "categorical_crossentropy"
        model, history = train(xtrain, y_train, xtest, y_test,
                               input_shape=(max_seq_len, n_features),
                               n_classes=n_classes,
                               drop_out=drop_out,
                               special_value=pad_value,
                               lr=lr,
                               epochs=epochs,
                               batch_size=batch_size,
                               loss_func=loss_func,
                               lstm_units=lstm_units)

        predictions = predict(model, xtest)
        where_to_write = "/mnt/shared/people/masoumeh/MA/results/plots/"
        error_analysis(history=history,
                       y_true=ytest,
                       y_pred=predictions,
                       model_name='lstm',
                       set_name=fname_prefix,
                       task='multi',
                       iters=iters,
                       path_to_save=where_to_write)
        print('-------------------------------------------')
        print()
        target_names=['deictic', 'sequential', 'demarcative']
        report = classification_report(ytest, predictions, target_names=target_names)
        print(report)
        print()
        return compute_acc_prec_rec_f1(y_true=ytest, y_pred=predictions, average=metric_avg)


a_scores = list()
p_scores = list()
r_scores = list()
f_scores = list()
for i in range(1):
    print('iteration: ', i)
    print()
    acc, per, rec, f1 = run(fname_prefix='set_6', iters=i, metric_avg="weighted", lstm_units=50)
    a_scores.append(acc)
    p_scores.append(per)
    r_scores.append(rec)
    f_scores.append(f1)

print('mean acc: ', mean(a_scores))
print('std acc: ', np.array(a_scores).std())
print('----------------------------------')
print('mean precision: ', mean(p_scores))
print('std precision: ', np.array(p_scores).std())
print('----------------------------------')
print('mean recall: ', mean(r_scores))
print('std recall: ', np.array(r_scores).std())
print('----------------------------------')
print('mean f1: ', mean(f_scores))
print('std f1: ', np.array(f_scores).std())
print('----------------------------------')


