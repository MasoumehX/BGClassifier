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


def precision_macro(y_true, y_pred):
    return precision_score(y_true, y_pred, average='macro')


def recall_macro(y_true, y_pred):
    return recall_score(y_true, y_pred, average='macro')


def f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')


def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def compute_acc_acc5_f1_prec_rec(y_true, y_pred):
    dic_model = dict()
    dic_model['acc'] = accuracy(y_true, y_pred)
    dic_model['precision_macro'] = precision_macro(y_true, y_pred)
    dic_model['recall_macro'] = recall_macro(y_true, y_pred)
    dic_model['f1-macro'] = f1_macro(y_true, y_pred)
    print('Accuracy: ', "%.2f" % (dic_model["acc"] * 100))
    print('precision macro: ', "%.2f" % (dic_model["precision_macro"] * 100))
    print('recall macro: ', "%.2f" % (dic_model['recall_macro'] * 100))
    print('F1 macro: ', "%.2f" % (dic_model['f1-macro'] * 100))
    return dic_model