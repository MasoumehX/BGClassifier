
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from keras import backend as K


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def precision_measure(y_true, y_pred, average='macro'):
    return precision_score(y_true, y_pred, average=average)


def recall_measure(y_true, y_pred, average='macro'):
    return recall_score(y_true, y_pred, average=average)


def f1_measure(y_true, y_pred, average='macro'):
    return f1_score(y_true, y_pred, average=average)


def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def compute_acc_prec_rec_f1(y_true, y_pred, average='macro'):
    acc = accuracy(y_true, y_pred)
    precision = precision_measure(y_true, y_pred, average=average)
    recall = recall_measure(y_true, y_pred, average=average)
    f1 = f1_measure(y_true, y_pred, average=average)
    print('accuracy: ', "%.2f" % (acc * 100))
    print('precision: ', "%.2f" % (precision * 100))
    print('recall: ', "%.2f" % (recall * 100))
    print('f1: ', "%.2f" % (f1 * 100))
    return acc, precision, recall, f1






