import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


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